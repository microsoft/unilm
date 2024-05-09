#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""
import models
import tasks
import criterions

import argparse
import logging
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from fairseq import checkpoint_utils, options, quantization_utils, tasks, utils
from fairseq.data import data_utils, iterators
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap
from fairseq.distributed import utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar


def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    if (
        distributed_utils.is_master(cfg.distributed_training)
        and "job_logging_cfg" in cfg
    ):
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    if cfg.common.log_file is not None:
        handler = logging.FileHandler(filename=cfg.common.log_file)
        logger.addHandler(handler)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    # if distributed_utils.is_master(cfg.distributed_training):
    #     checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logging.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(cfg.distributed_training):
            model = fsdp_wrap(task.build_model(cfg.model))
    else:
        model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)

    tpu = cfg.common.tpu
    cuda = torch.cuda.is_available() and not cfg.common.cpu and not tpu
    if cuda:
        device = torch.device("cuda")
    elif tpu:
        device = utils.get_tpu_device()
    else:
        device = torch.device("cpu")
    if cfg.common.fp16:
        criterion = criterion.half()
        model = model.half()
    elif cfg.common.bf16:
        criterion = criterion.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)
    criterion = criterion.to(device)
    model = model.to(device)

    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(
                p.numel() for p in model.parameters() if not getattr(p, "expert", False)
            ),
            sum(
                p.numel()
                for p in model.parameters()
                if not getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )

    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(
                p.numel()
                for p in model.parameters()
                if getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # We load the valid dataset AFTER building the model
    data_utils.raise_if_valid_subsets_unintentionally_ignored(cfg)
    if cfg.dataset.combine_valid_subsets:
        task.load_dataset("valid", combine=True, epoch=1)
    else:
        for valid_sub_split in cfg.dataset.valid_subset.split(","):
            task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    # try:
    #     state_dict = torch.load(cfg.checkpoint.restore_file)
    #     model.load_state_dict(state_dict['model'])
    #     print(f"Loaded model from {cfg.checkpoint.restore_file}")
    # except Exception as e:
    #     print(e)
    #     print(f"No checkpoint found from {cfg.checkpoint.restore_file}")

    valid_subsets = cfg.dataset.valid_subset.split(",")
    logger.info("Start validating")

    validate(
        cfg, task, model, criterion, valid_subsets,
    )

@torch.no_grad()
def validate(
    cfg: DictConfig,
    task: tasks.FairseqTask,
    model,
    criterion,
    subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""
    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    valid_losses = []
    for subset in subsets:
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=cfg.dataset.max_tokens_valid,
            max_sentences=cfg.dataset.batch_size_valid,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_workers=cfg.dataset.num_workers_valid,
            # always pass a fixed "epoch" to keep validation data consistent
            # across training epochs
            epoch=1,
            data_buffer_size=cfg.dataset.data_buffer_size,
        ).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        )
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            logging_outputs = []
            for i, sample in enumerate(progress):
                if cfg.dataset.max_valid_steps is not None and i > cfg.dataset.max_valid_steps:
                    break
                sample = utils.move_to_cuda(sample)
                _, _, inner_logging_outputs = task.valid_step(
                    sample, model, criterion
                )
                logging_outputs.append(inner_logging_outputs)
            task.reduce_metrics(logging_outputs, criterion)
            
        # with metrics.aggregate(new_root=True) as agg:
        #     for i, sample in enumerate(progress):
        #         if (
        #             cfg.dataset.max_valid_steps is not None
        #             and i > cfg.dataset.max_valid_steps
        #         ):
        #             break
        #         trainer.valid_step(sample)

        stats = get_valid_stats(cfg, agg.get_smoothed_values())
        
        progress.print(stats, tag=subset)

        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(
    cfg: DictConfig, stats: Dict[str, Any]
) -> Dict[str, Any]:
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(
            f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}"
        )

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

    # if cfg.common.use_plasma_view:
    #     server.server.kill()


if __name__ == "__main__":
    cli_main()