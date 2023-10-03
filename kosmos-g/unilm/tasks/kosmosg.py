# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
from argparse import Namespace
from dataclasses import dataclass, field

import torch
from deepspeed.runtime.engine import DeepSpeedEngine

from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.tasks import register_task
from unilm.data.basic_loader import MixLoader
from unilm.data.vl.instructpix2pix_loader import InstructPix2PixLoader
from unilm.data.vl.laion2b_loader import Laion2BLoader
from unilm.data.vl.openimage_loader import OpenImageLoader
from unilm.data.vl.vl_loader import WdsLoaderConfig
from unilm.tasks.gpt_base import GPTPretrainingConfig, GPTTask

logger = logging.getLogger(__name__)


@dataclass
class KosmosGConfig(GPTPretrainingConfig, WdsLoaderConfig):
    max_image_num: int = field(default=5, metadata={"help": ""})
    image_token_length: int = field(default=64, metadata={"help": ""})
    laion_data_dir: str = field(default="", metadata={"help": ""})
    laion_batch_size: int = field(default=1, metadata={"help": ""})
    instructpix2pix_data_dir: str = field(default="", metadata={"help": ""})
    instructpix2pix_batch_size: int = field(default=1, metadata={"help": ""})
    openimage_data_dir: str = field(default="", metadata={"help": ""})
    openimage_batch_size: int = field(default=1, metadata={"help": ""})
    data_weights: str = field(
        default="1,2,2",
        metadata={"help": "laion, instructpix2pix, openimage"},
    )
    caption_dropout_prob: float = field(default=0.3, metadata={"help": ""})
    align: bool = field(default=False, metadata={"help": ""})
    random_drop_caption_prob: float = field(default=0.0, metadata={"help": ""})


@register_task("kosmosg", dataclass=KosmosGConfig)
class KosmosGTask(GPTTask):
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
        laion_dataset = Namespace(**{
            'data': json.load(open(f'{self.cfg.laion_data_dir}/json/train.json')),
            'data_dir': self.cfg.laion_data_dir,
            'shuffle': True})

        lain_vl_loader = Laion2BLoader(
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

        instructpix2pix_dataset = Namespace(**{
            'data': json.load(open(f'{self.cfg.instructpix2pix_data_dir}/json/train.json')),
            'data_dir': self.cfg.instructpix2pix_data_dir,
            'shuffle': True})

        instructpix2pix_vl_loader = InstructPix2PixLoader(
            self.cfg,
            instructpix2pix_dataset,
            self.dictionary,
            self.tokenizer,
            max_tokens=max_tokens,
            max_sentences=self.cfg.instructpix2pix_batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=ignore_invalid_inputs,
            required_batch_size_multiple=required_batch_size_multiple,
            seed=seed,
            epoch=epoch,
            num_shards=num_shards,
            shard_id=shard_id,
            no_prefetch=False,
        )

        openimage_dataset = Namespace(**{
            'data': json.load(open(f'{self.cfg.openimage_data_dir}/json/train.json')),
            'data_dir': self.cfg.openimage_data_dir,
            'shuffle': True})

        openimage_vl_loader = OpenImageLoader(
            self.cfg,
            openimage_dataset,
            self.dictionary,
            self.tokenizer,
            max_tokens=max_tokens,
            max_sentences=self.cfg.openimage_batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=ignore_invalid_inputs,
            required_batch_size_multiple=required_batch_size_multiple,
            seed=seed,
            epoch=epoch,
            num_shards=num_shards,
            shard_id=shard_id,
            no_prefetch=False,
        )

        data_weight = [float(x) for x in self.cfg.data_weights.split(',')]
        data_weight = [x / sum(data_weight) for x in data_weight]

        concat_loader = MixLoader([
            lain_vl_loader,
            instructpix2pix_vl_loader,
            openimage_vl_loader,
        ], data_weight)
        return concat_loader

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        self.datasets[split] = {
            'data': None,
            'data_dir': None,
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

        if 'vl_laion' in sample:
            loss_name = "image_laion"
            loss_key = 'vl_laion'
        elif 'vl_instructpix2pix' in sample:
            loss_name = "image_instructpix2pix"
            loss_key = 'vl_instructpix2pix'
        elif 'vl_openimage' in sample:
            loss_name = "image_openimage"
            loss_key = 'vl_openimage'
        else:
            assert False, "Unknown loss key"

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
