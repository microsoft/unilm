from dataclasses import dataclass, field
from typing import Optional
import logging
from argparse import Namespace
import json
import torch

from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.tasks import FairseqTask, register_task
# from unilm.data.lm_loader import LMLoader
from unilm.data.spm_lm_loader import SpmLmLoader as LMLoader
from unilm.tasks.gpt_base import GPTPretrainingConfig, GPTTask
from unilm.data.basic_loader import MixLoader

from unilm.data.vl.vl_loader import WdsLoaderConfig

from unilm.data.vl.Interleaved_loader import InterleavedLoader
from unilm.data.vl.laion2b_loader import Laion2BLoader
from unilm.data.vl.laion2b_obj_loader import Laion2BObjLoader
from deepspeed.runtime.engine import DeepSpeedEngine

logger = logging.getLogger(__name__)


@dataclass
class ImageGPTPretrainingConfig(GPTPretrainingConfig, WdsLoaderConfig):
    max_image_num: int = field(default=5, metadata={"help": ""})
    image_token_length: int = field(default=64, metadata={"help": ""})
    interleaved_data_dir: str = field(default="", metadata={"help": ""})
    interleaved_batch_size: int = field(default=4, metadata={"help": ""})
    laion_data_dir: str = field(default="", metadata={"help": ""})
    laion_batch_size: int = field(default=32, metadata={"help": ""})
    data_weights: str = field(default="0,32,1", metadata={"help": "interleaved,laion,gpt"})
    input_resolution: int = field(default=224, metadata={"help": ""})
    # newsetting
    quantized_size: int = field(default=16, metadata={"help": "used to discrete the continuous coordinates"})
    locate_special_token: int = field(default=1, metadata={"help": "used special token (grounding) reprsent need to ouput bbox"})
    phrase_mode: str = field(default="expression", metadata={"help": "mode in phrase,expression"})
    training_image_only_resize: int = field(default=0, metadata={"help": "only use resize transform during pretraining"})
    
    # some parameters to filter the bounding box used for pretraining
    box_score_threshold: float = field(default=0.65, metadata={"help": "filter the box with low confidence"})
    mix_no_object_prob: float = field(default=0., metadata={"help": "prob of using the image-text pairs that w/o box"})
    use_object_bbox_prob: float = field(default=1., metadata={"help": "prob of using the image-text pairs that w box"})
    
@register_task("image_gpt_interleaved_laion_obj", dataclass=ImageGPTPretrainingConfig)
class ImageGPTInterleavedLaionV3ObjTask(GPTTask):
    def __init__(self, cfg, dictionary, tokenizer):
        super().__init__(cfg, dictionary, tokenizer)
        self.vlm_model = None

    def build_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint=from_checkpoint)
        self.vlm_model = model
        return model

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False
    ):  
        interleaved_dataset = Namespace(**{
            'data': json.load(open(f'{self.cfg.interleaved_data_dir}/json/train.json')),
            'data_dir': self.cfg.interleaved_data_dir,
            'shuffle': True})

        vl_loader = InterleavedLoader(
            self.cfg,
            interleaved_dataset,
            self.dictionary,
            self.tokenizer,
            max_tokens=max_tokens,
            max_sentences=self.cfg.interleaved_batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=ignore_invalid_inputs,
            required_batch_size_multiple=required_batch_size_multiple,
            seed=seed,
            epoch=epoch,
            num_shards=num_shards,
            shard_id=shard_id,
            no_prefetch=False,
        )

        laion_dataset = Namespace(**{
            'data': json.load(open(f'{self.cfg.laion_data_dir}/json/train.json')),
            'data_dir': self.cfg.laion_data_dir,
            'shuffle': True})

        lain_vl_loader = Laion2BObjLoader(
            self.cfg,
            laion_dataset,
            self.dictionary,
            self.tokenizer,
            max_tokens=max_tokens,
            max_sentences=self.cfg.laion_batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=ignore_invalid_inputs,
            required_batch_size_multiple=required_batch_size_multiple,
            seed=seed,
            epoch=epoch,
            num_shards=num_shards,
            shard_id=shard_id,
            no_prefetch=False,
        )

        lm_loader = LMLoader(
            self.cfg,
            dataset,
            self.dictionary,
            self.tokenizer,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            max_positions=max_positions,
            ignore_invalid_inputs=ignore_invalid_inputs,
            required_batch_size_multiple=required_batch_size_multiple,
            seed=seed,
            epoch=epoch,
            num_shards=num_shards,
            shard_id=shard_id,
        )

        data_weight = [float(x) for x in self.cfg.data_weights.split(',')]
        data_weight = [x / sum(data_weight) for x in data_weight]

        concat_loader = MixLoader([vl_loader, lain_vl_loader, lm_loader], data_weight)
        return concat_loader

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        json_split_name = self.cfg.train_json_split_name if split == "train" else split
        self.datasets[split] = {
            'data': json.load(open(f'{self.cfg.data}/json/{json_split_name}.json')),
            'data_dir': self.cfg.data,
            'shuffle': True if split == 'train' else False, }
        self.datasets[split] = Namespace(**self.datasets[split])

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)

        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}

        loss_name = None
        loss_key = None
        if 'vl' in sample:
            loss_name = "image_interleaved"
            loss_key = 'vl'
        elif 'vl_laion' in sample:
            loss_name = "image_laion"
            loss_key = 'vl_laion'
        elif 'gpt' in sample:
            loss_name = "gpt"
            loss_key = 'gpt'
        else:
            assert False, "Unknown loss key"

        # pdb.set_trace()
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                vl_loss, sample_size, logging_output = criterion(model, sample[loss_key], loss_name=loss_name)
        if ignore_grad:
            vl_loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            if isinstance(model, DeepSpeedEngine):
                model.backward(vl_loss)
            else:
                optimizer.backward(vl_loss)
        
        agg_loss += vl_loss.detach().item()
        agg_sample_size += sample_size
        agg_logging_output.update(logging_output)

        return agg_loss, agg_sample_size, agg_logging_output

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

