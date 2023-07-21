import logging
import math
try:
    import webdataset as wds
except ImportError:
    print('Please install webdataset: pip install webdataset for VL dataset')

from argparse import Namespace
from dataclasses import dataclass, field
from fairseq.dataclass import ChoiceEnum, FairseqDataclass

try:
    from unilm.data.vl.clip_dl import (
        DataInfo,
        detshuffle2,
        filter_no_caption,
        get_dataset_size,
        log_and_continue,
        ResampledShards2,
        _SAMPLE_SHUFFLE_SIZE,
        _SAMPLE_SHUFFLE_INITIAL,
        _SHARD_SHUFFLE_SIZE,
        _SHARD_SHUFFLE_INITIAL,
        SharedEpoch, 
        tarfile_to_samples_nothrow, )
except ImportError:
    print('Please install pip install -r visual_requirement.txt for VL dataset')


logger = logging.getLogger(__name__)


def preprocess_txt(text):
    return str(text)


@dataclass
class WdsLoaderConfig(FairseqDataclass):
    wds_train_data: str = field(default="", metadata={"help": "wds train data"})
    wds_val_data: str = field(default="", metadata={"help": "wds val data"})
    wds_dataset_resampled: bool = field(default=False, metadata={"help": ""})
    wds_train_num_samples: int = field(default=0, metadata={"help": ""})
    wds_val_num_samples: int = field(default=0, metadata={"help": ""})
    wds_format: str = field(default="20m1k", metadata={"help": "20m1k or laion"})
    # wds_tokens_per_sample: int = field(default=77, metadata={"help": ""})


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, shard_id=0, num_shards=1, max_sentences=None):
    wds_args = Namespace(
        train_data=args.wds_train_data,
        val_data=args.wds_val_data,
        dataset_resampled=args.wds_dataset_resampled,
        train_num_samples=args.wds_train_num_samples,
        val_num_samples=args.wds_val_num_samples,
        seed=args.seed + shard_id,
        batch_size=max_sentences,
        workers=1,
        world_size=num_shards, )
    return _get_wds_dataset(wds_args, preprocess_img, is_train, epoch=epoch, floor=floor)


def _get_wds_dataset(wds_args, preprocess_img, is_train, epoch=0, floor=False):
    args = wds_args
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0    # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)    # create a shared epoch store to sync epoch to dataloader worker proc
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch, ),
                wds.split_by_node,
                wds.split_by_worker,])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,    # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL, ), ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),])
    pipeline.extend([
        wds.select(filter_no_caption),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png", text="txt"),
        wds.map_dict(image=preprocess_img, text=preprocess_txt),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train),])

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)    # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)    # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)