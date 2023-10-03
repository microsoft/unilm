"""
DeepSpeed trainer
"""

import os
import sys
import torch
import time
import logging
import deepspeed
import json
import subprocess

from typing import Any, Dict, List
from itertools import chain
from argparse import Namespace
import torch.distributed as dist

from fairseq import checkpoint_utils, models, optim, utils
from fairseq.distributed import utils as distributed_utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.logging import meters, metrics
from fairseq.optim import lr_scheduler
from fairseq.optim.dynamic_loss_scaler import DynamicLossScaler
from fairseq.trainer import Trainer
from fairseq.file_io import PathManager

from omegaconf import OmegaConf


logger = logging.getLogger(__name__)


def get_config(config, full_name, fairseq_value):
    _config = config
    for name in full_name.split(":"):
        if name in _config:
            _config = _config[name]
        else:
            _config = fairseq_value
            break
    assert _config == fairseq_value, f"deepspeed config: {full_name} does not align with fairseq value: {fairseq_value}"
    return _config


class DeepSpeedTrainer(object):
    """Main class for data parallel training w. DeepSpeed.

    Similar to fairseq.Trainer this class supports synchronous distributed 
    data parallel training. However, in this case we expose DeepSpeed features
    like ZeRO stages 1, 2, and 3.
    """

    def __init__(self, cfg: FairseqConfig, task, model, criterion):
        if isinstance(cfg, Namespace):
            logger.warning(
                "argparse.Namespace configuration is deprecated! Automatically converting to OmegaConf"
            )
            cfg = convert_namespace_to_omegaconf(cfg)

        self.cfg = cfg
        self.task = task

        # try:
        #      subprocess.check_output('which rsync', shell=True)
        # except subprocess.CalledProcessError:
        #     raise RuntimeError('Please install rsync, this is required for model checkpointing')

        ds_config = {}
        if isinstance(cfg.common.deepspeed, str):
            assert os.path.isfile(cfg.common.deepspeed), f"deepspeed config path is not a file: {cfg.common.deepspeed}"
            with open(cfg.common.deepspeed, 'r') as fd:
                ds_config = json.load(fd)

        # ds_config['zero_allow_untested_optimizer'] = True

        # gradient accumulation steps
        assert len(self.cfg.optimization.update_freq) == 1, "no support for gradient accumulation schedules"
        gas = ds_config.get("gradient_accumulation_steps", self.cfg.optimization.update_freq[0])
        ds_config["gradient_accumulation_steps"] = gas

        # train_micro_batch_size_per_gpu
        micro_batch_size = get_config(ds_config, "train_micro_batch_size_per_gpu", self.cfg.dataset.batch_size)
        # micro_batch_size = get_config(ds_config, "train_micro_batch_size_per_gpu", self.cfg.dataset.max_tokens // self.cfg.task.tokens_per_sample)
        ds_config["train_micro_batch_size_per_gpu"] = int(micro_batch_size)

        # enable fp16
        fp16 = get_config(config=ds_config, full_name="fp16:enabled", fairseq_value=self.cfg.common.fp16)
        if "fp16" not in ds_config:
            ds_config["fp16"] = {}
        ds_config["fp16"]["enabled"] = fp16

        # gradient_clipping self.cfg.optimization.clip_norm
        grad_clip = get_config(ds_config, "gradient_clipping", self.cfg.optimization.clip_norm)
        ds_config["gradient_clipping"] = grad_clip
        
        # force zero elastic checkpoint disabled
        elastic_ckpt = get_config(ds_config, "zero_optimization:elastic_checkpoint", False)
        if "zero_optimization" not in ds_config:
            ds_config["zero_optimization"] = {}
        ds_config["zero_optimization"]["elastic_checkpoint"] = elastic_ckpt

        zero_stage = get_config(ds_config, "zero_optimization:stage", cfg.common.zero)
        ds_config["zero_optimization"]["stage"] = zero_stage
        self.zero_stage = int(zero_stage)

        ds_config["zero_optimization"]["contiguous_gradients"] = False

        assert cfg.common.zero != 3, "zero stage 3 is currently untested with this codebase"

        self.ds_config = ds_config

        print(f"****** fairseq generated ds-config: {self.ds_config}")

        # catalog shared parameters
        shared_params = _catalog_shared_params(model)
        self.tpu = cfg.common.tpu
        assert not self.tpu, "deepspeed does not support tpu"
        self.cuda = torch.cuda.is_available()
        assert self.cuda, "deepspeed assumes cuda devices are available"
        self.device = torch.device("cuda")

        self._criterion = criterion
        self._model = model
        
        # assert self.cfg.common.fp16, "only fp16 is supported"
        assert self.cfg.distributed_training.ddp_backend in ["c10d", "legacy_ddp", "no_c10d"]
        assert cfg.distributed_training.ddp_backend != "fully_sharded"
        assert not cfg.common.bf16, "bf16 not yet supported"
        assert not cfg.distributed_training.pipeline_model_parallel, "pipeline not yet supported"
        assert not self.cfg.optimization.use_bmuf, "bmuf not yet supported"
        assert self.cfg.distributed_training.zero_sharding != "os"
        assert not self.cfg.common.memory_efficient_fp16, "mem efficient fp16 not yet supported"
        assert self.cfg.distributed_training.ddp_backend != "slow_mo"

        self.pipeline_model_parallel = cfg.distributed_training.pipeline_model_parallel

        self.zero_enabled = False

        if cfg.common.fp16:
            self._criterion = self._criterion.half()
            self._model = self._model.half()

        assert not utils.has_parameters(self._criterion), "criterion has params, not supported yet"

        # check that shared parameters are preserved after device transfer
        for shared_param in shared_params:
            ref = _get_module_by_path(self._model, shared_param[0])
            for path in shared_param[1:]:
                logger.info(
                    "detected shared parameter: {} <- {}".format(shared_param[0], path)
                )
                _set_module_by_path(self._model, path, ref)

        self._dummy_batch = None  # indicates we don't have a dummy batch at first
        self._lr_scheduler = None
        self._num_updates = 0
        self._num_xla_compiles = 0  # for TPUs
        self._optim_history = None
        self._optimizer = None
        self._warn_once = set()
        self._wrapped_criterion = None
        self._wrapped_model = None

        self.train_step_count = 0

        if self.cuda and self.data_parallel_world_size > 1:
            self._grad_norm_buf = torch.cuda.DoubleTensor(self.data_parallel_world_size)
        else:
            self._grad_norm_buf = None

        self.quantizer = None

        # get detailed cuda environment
        if self.cuda:
            self.cuda_env = utils.CudaEnvironment()
            if self.data_parallel_world_size > 1:
                self.cuda_env_arr = distributed_utils.all_gather_list(
                    self.cuda_env, group=distributed_utils.get_global_group()
                )
            else:
                self.cuda_env_arr = [self.cuda_env]
            if self.data_parallel_rank == 0:
                utils.CudaEnvironment.pretty_print_cuda_env_list(self.cuda_env_arr)
        else:
            self.cuda_env = None
            self.cuda_env_arr = None

        metrics.log_start_time("wall", priority=790, round=2)

        self._start_time = time.time()
        self._previous_training_time = 0
        self._cumulative_training_time = None

        self._build_optimizer()

    def _build_optimizer(self):
        params = list(
            filter(
                lambda p: p.requires_grad,
                chain(self.model.parameters(), self.criterion.parameters()),
            )
        )

        # create simple optimizer, DS will handle fp16 wrappers
        optimizer = optim.build_optimizer(self.cfg.optimizer, params)

        print(f"************ built fairseq optimizer: {optimizer}")
    
        os.environ['LOCAL_RANK'] = str(self.cfg.distributed_training.device_id)
        os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = str(self.cfg.distributed_training.device_id)
        self.device = torch.device("cuda", self.cfg.distributed_training.device_id)
        self.model.to(device=self.device)

        torch.distributed.barrier()
        print(f'done pre-engine rank={dist.get_rank()}')

        engine, optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer,
            config_params=self.ds_config
        )

        self.zero_enabled = engine.zero_optimization_stage() > 0

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.
        self._lr_scheduler = lr_scheduler.build_lr_scheduler(
            self.cfg.lr_scheduler,
            engine.optimizer,
        )
        self._lr_scheduler.step_update(0)

        self._optimizer = optimizer
        self._wrapped_model = engine
        self.device = engine.device
        self._criterion.to(device=self.device)
        print(f"local_rank={torch.distributed.get_rank()}, engine.device={engine.device}") #, engine.module.device={engine.module.device}")
        torch.distributed.barrier()

        if getattr(self.cfg.common, "fp16_scale_window", None) is None:
            if len(self.cfg.optimization.update_freq) > 1:
                raise ValueError(
                    "--fp16-scale-window must be given explicitly when using a "
                    "custom --update-freq schedule"
                )
            data_parallel_size = int(
                self.cfg.distributed_training.distributed_world_size
                / self.cfg.common.model_parallel_size
            )
            scale_window = int(
                2 ** 14 / data_parallel_size / self.cfg.optimization.update_freq[0]
            )
        else:
            scale_window = self.cfg.common.fp16_scale_window

        self.scaler = DynamicLossScaler(
            init_scale=self.cfg.common.fp16_init_scale,
            scale_window=scale_window,
            tolerance=self.cfg.common.fp16_scale_tolerance,
            threshold=self.cfg.common.threshold_loss_scale,
            min_loss_scale=self.cfg.common.min_loss_scale,
        )

    @metrics.aggregate("train")
    def train_step(self, samples, raise_oom=False):
        """Do forward, backward and parameter update."""
        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        metrics.log_start_time("train_wall", priority=800, round=2)

        grad_norm = torch.tensor(0.0).cuda()

        # let fairseq handle grad accumlation scaling
        self.model.scale_wrt_gas = False

        # forward and backward pass
        logging_outputs, sample_size, ooms = [], 0, 0
        # print(f'** samples size: {len(samples)}')
        sample_count = len(samples)
        for i, sample in enumerate(samples):  # delayed update loop
            sample, is_dummy_batch = self._prepare_sample(sample)

            if self.cfg.common.fp16:
                self.model.optimizer.override_loss_scale(self.scaler.loss_scale)

            self.model.set_gradient_accumulation_boundary(is_boundary=False)
            try:
                # forward and backward
                # print(f'i={i}, rank={dist.get_rank()}, pre task.train_step')
                dist.barrier()
                loss, sample_size_i, logging_output = self.task.train_step(
                    sample=sample,
                    model=self.model,
                    criterion=self.criterion,
                    optimizer=self.optimizer,
                    update_num=self.get_num_updates(),
                    ignore_grad=is_dummy_batch,
                )
                self.train_step_count += 1
                
                # increment deepspeed micro step on non-final train step since optimizer.step will increment it for us
                if (i + 1) != sample_count:
                    self.model.micro_steps += 1

                self.model.set_gradient_accumulation_boundary(is_boundary=True)
                if self.zero_stage <= 2:
                    self.model.allreduce_gradients()

                # print(f'grads[0]={list([p.grad for p in self.model.optimizer.fp16_groups[0]])}')
                # print(f'train_step={self.train_step_count}, loss_scale={self.model.optimizer.cur_scale}')
                # print(f'i={i}, rank={dist.get_rank()}, loss={loss}')
                if torch.distributed.get_rank() == 0:
                    _loss_scale = self.model.optimizer.external_loss_scale if self.cfg.common.fp16 else 0
                    print(f"[{torch.distributed.get_rank()}], " \
                            f"micro_step={self.model.micro_steps}, " \
                            f"gas_boundary={self.model.is_gradient_accumulation_boundary()}, " \
                            f"train_step={self.train_step_count}, " \
                            f"lr={self.get_lr()}, " \
                            f"loss_scale={_loss_scale}, " \
                            f"loss={loss}")

                if self.cfg.common.exit_interval and self.train_step_count % self.cfg.common.exit_interval == 0:
                    if torch.distributed.get_rank() == 0:
                        print("exiting early...")
                    sys.exit()

                logging_outputs.append(logging_output)
                sample_size += sample_size_i

                # emptying the CUDA cache after the first step can
                # reduce the chance of OOM
                if self.cuda and self.get_num_updates() == 0:
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if raise_oom:
                        raise e
                    logger.warning(
                        "attempting to recover from OOM in forward/backward pass"
                    )
                    ooms += 1
                    self.zero_grad()
                    if self.cuda:
                        torch.cuda.empty_cache()
                    if self.cfg.distributed_training.distributed_world_size == 1:
                        return None
                else:
                    raise e

        if is_dummy_batch:
            if torch.is_tensor(sample_size):
                sample_size.zero_()
            else:
                sample_size *= 0.0

        if torch.is_tensor(sample_size):
            sample_size = sample_size.float()
        else:
            sample_size = float(sample_size)

        # gather logging outputs from all replicas
        if self.data_parallel_world_size > 1:
            train_time = self._local_cumulative_training_time()
            logging_outputs, (
                sample_size,
                ooms,
                total_train_time,
            ) = self._aggregate_logging_outputs(
                logging_outputs, sample_size, ooms, train_time, ignore=is_dummy_batch
            )
            self._cumulative_training_time = (
                total_train_time / self.data_parallel_world_size
            )

        # grad_norms = []
        # for param in self.optimizer.fp16_params:
        #     if param.grad is not None:
        #         grad_norms.append(param.grad.norm())
        # print(grad_norms)

        # self.model.optimizer.dump_grad_norms()

        ######################
        ### multiply grads ###
        ######################
        # numer = (
        #     self.data_parallel_world_size
        #     if not self.cfg.optimization.use_bmuf or self._sync_stats()
        #     else 1
        # )
        # self.optimizer.multiply_grads(numer / (sample_size or 1.0))
        # grad_norm = self.optimizer.clip_grad_norm(self.cfg.optimization.clip_norm)

        overflow = False
        try:
            with torch.autograd.profiler.record_function("optimizer"):
                # take an optimization step
                self.task.optimizer_step(
                    self.optimizer, model=self.model, update_num=self.get_num_updates()
                )
                # pass overflow flag from ds to fairseq
                if self.cfg.common.fp16:
                    overflow = self.model.optimizer.overflow
                    self.scaler.check_overflow(overflow=overflow)
                    self.scaler.update()

                _grad_norm = self.model.get_global_grad_norm()
                if _grad_norm is not None:
                    grad_norm = torch.tensor(_grad_norm).to(self.device)
        except FloatingPointError:
            raise
        except OverflowError as e:
            logger.info(f"NOTE: gradient overflow detected, ignoring gradient, {str(e)}")
            overflow = True
        
        # import sys; sys.exit()
        # except RuntimeError as e:
        #     raise e

        # except FloatingPointError:
        #     # re-run the forward and backward pass with hooks attached to print
        #     # out where it fails
        #     self.zero_grad()
        #     with NanDetector(self.get_model()):
        #         for _, sample in enumerate(samples):
        #             sample, _ = self._prepare_sample(sample)
        #             self.task.train_step(
        #                 sample,
        #                 self.model,
        #                 self.criterion,
        #                 self.optimizer,
        #                 self.get_num_updates(),
        #                 ignore_grad=False,
        #             )
        #     raise
        # except OverflowError as e:
        #     overflow = True
        #     logger.info(f"NOTE: gradient overflow detected, ignoring gradient, {str(e)}")
        #     grad_norm = torch.tensor(0.0).cuda()
        #     self.zero_grad()
        # except RuntimeError as e:
        #     if "out of memory" in str(e):
        #         self._log_oom(e)
        #         logger.error("OOM during optimization, irrecoverable")
        #     raise e

        # Some distributed wrappers (e.g., SlowMo) need access to the optimizer
        # after the step
        # if hasattr(self.model, "perform_additional_optimizer_actions"):
        #     if hasattr(self.optimizer, "fp32_params"):
        #         self.model.perform_additional_optimizer_actions(
        #             self.optimizer.optimizer, self.optimizer.fp32_params
        #         )
        #     else:
        #         self.model.perform_additional_optimizer_actions(
        #             self.optimizer.optimizer
        #         )

        logging_output = None
        if not overflow or self.cfg.distributed_training.ddp_backend == "slow_mo":
            self.set_num_updates(self.get_num_updates() + 1)

            if self.cuda and self.cuda_env is not None:
                # log minimum free memory over the iteration
                gb_used = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
                torch.cuda.reset_peak_memory_stats()
                gb_free = self.cuda_env.total_memory_in_GB - gb_used
                metrics.log_scalar(
                    "gb_free", gb_free, priority=1500, round=1, weight=0
                )

            # log stats
            logging_output = self._reduce_and_log_stats(
                logging_outputs, sample_size, grad_norm
            )

            # clear CUDA cache to reduce memory fragmentation
            # if (
            #     self.cuda
            #     and self.cfg.common.empty_cache_freq > 0
            #     and (
            #         (self.get_num_updates() + self.cfg.common.empty_cache_freq - 1)
            #         % self.cfg.common.empty_cache_freq
            #     )
            #     == 0
            # ):
            #     torch.cuda.empty_cache()

        if self.cfg.common.fp16:
            metrics.log_scalar(
                "loss_scale",
                # self.optimizer.loss_scaler.loss_scaler,
                self.optimizer.cur_scale,
                priority=700,
                round=4,
                weight=0,
            )

        metrics.log_stop_time("train_wall")
        return logging_output

    def _sync_stats():
        if self.data_parallel_world_size == 1:
            return False
        else:
            return True

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        self.lr_step_update()
        metrics.log_scalar("num_updates", self._num_updates, weight=0, priority=200)

    def _aggregate_logging_outputs(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        if self.task.__class__.logging_outputs_can_be_summed(self.get_criterion()):
            return self._fast_stat_sync_sum(
                logging_outputs, *extra_stats_to_sum, ignore=ignore
            )
        else:
            return self._all_gather_list_sync(
                logging_outputs, *extra_stats_to_sum, ignore=ignore
            )

    def _fast_stat_sync_sum(
        self, logging_outputs: List[Dict[str, Any]], *extra_stats_to_sum, ignore=False,
    ):
        """
        Sync logging outputs across workers. fast_stat_sync_sum is
        faster than all_gather_list_sync, but is only suitable when
        logging outputs are scalars and can be summed. Note that
        *logging_outputs* cannot contain any nested dicts/lists.
        """
        data = {}
        for i, stat in enumerate(extra_stats_to_sum):
            data["extra_stats_" + str(i)] = stat
        if len(logging_outputs) > 0:
            log_keys = list(logging_outputs[0].keys())
            for k in log_keys:
                if not ignore:
                    v = sum(log[k] for log in logging_outputs if k in log)
                else:
                    v = logging_outputs[0][k]
                    v = torch.zeros_like(v) if torch.is_tensor(v) else 0
                data["logging_outputs_" + k] = v
        else:
            log_keys = None

        data = distributed_utils.all_reduce_dict(
            data, device=self.device, group=self.data_parallel_process_group
        )

        extra_stats_to_sum = [
            data["extra_stats_" + str(i)] for i in range(len(extra_stats_to_sum))
        ]
        if log_keys is not None:
            logging_outputs = [{k: data["logging_outputs_" + k] for k in log_keys}]
        else:
            logging_outputs = []
        return logging_outputs, extra_stats_to_sum

    def _all_gather_list_sync(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        """
        Sync logging outputs across workers. all_gather_list_sync is
        suitable when logging outputs are complex types.
        """
        if self.tpu:
            raise NotImplementedError
        if ignore:
            logging_outputs = []
        results = list(
            zip(
                *distributed_utils.all_gather_list(
                    [logging_outputs] + list(extra_stats_to_sum),
                    max_size=getattr(self.cfg.common, "all_gather_list_size", 16384),
                    group=self.data_parallel_process_group,
                )
            )
        )
        logging_outputs, extra_stats_to_sum = results[0], results[1:]
        logging_outputs = list(chain.from_iterable(logging_outputs))
        extra_stats_to_sum = [sum(s) for s in extra_stats_to_sum]
        return logging_outputs, extra_stats_to_sum

    def _reduce_and_log_stats(self, logging_outputs, sample_size, grad_norm=None):
        if grad_norm is not None and (
            not torch.is_tensor(grad_norm) or torch.isfinite(grad_norm)
        ):
            metrics.log_speed("ups", 1.0, priority=100, round=2)
            metrics.log_scalar("gnorm", grad_norm, priority=400, round=3)
            if self.cfg.optimization.clip_norm > 0:
                metrics.log_scalar(
                    "clip",
                    torch.where(
                        grad_norm > self.cfg.optimization.clip_norm,
                        grad_norm.new_tensor(100),
                        grad_norm.new_tensor(0),
                    ),
                    priority=500,
                    round=1,
                )

        with metrics.aggregate() as agg:
            if logging_outputs is not None:
                self.task.reduce_metrics(logging_outputs, self.get_criterion())
                del logging_outputs

            # extra warning for criterions that don't properly log a loss value
            if "loss" not in agg:
                if "loss" not in self._warn_once:
                    self._warn_once.add("loss")
                    logger.warning(
                        "Criterion.reduce_metrics did not log a 'loss' value, "
                        "which may break some functionality"
                    )
                metrics.log_scalar("loss", -1)

            # support legacy interface
            if self.tpu:
                logging_output = {}
            else:
                logging_output = agg.get_smoothed_values()
                logging_output["sample_size"] = sample_size
                for key_to_delete in ["ppl", "wps", "wpb", "bsz"]:
                    if key_to_delete in logging_output:
                        del logging_output[key_to_delete]
            return logging_output

    def cumulative_training_time(self):
        if self._cumulative_training_time is None:
            # single GPU
            return self._local_cumulative_training_time()
        else:
            return self._cumulative_training_time

    def _local_cumulative_training_time(self):
        """Aggregate training time in seconds."""
        return time.time() - self._start_time + self._previous_training_time

    def _prepare_sample(self, sample, is_dummy=False):
        #DS: untouched
        if sample == "DUMMY":
            raise Exception(
                "Trying to use an uninitialized 'dummy' batch. This usually indicates "
                "that the total number of batches is smaller than the number of "
                "participating GPUs. Try reducing the batch size or using fewer GPUs."
            )

        if sample is None or len(sample) == 0:
            assert (
                self._dummy_batch is not None and len(self._dummy_batch) > 0
            ), "Invalid dummy batch: {}".format(self._dummy_batch)
            sample, _ = self._prepare_sample(self._dummy_batch, is_dummy=True)
            return sample, True

        if self.cuda:
            if self.pipeline_model_parallel:
                if "target" in sample:
                    sample["target"] = utils.move_to_cuda(
                        sample["target"], device=self.last_device
                    )
            else:
                sample = utils.move_to_cuda(sample)
        elif self.tpu and is_dummy:
            # the dummy batch may not be on the appropriate device
            sample = utils.move_to_cuda(sample, device=self.device)

        def apply_half(t):
            if t.dtype is torch.float32:
                return t.half()
            return t

        def apply_bfloat16(t):
            if t.dtype is torch.float32:
                return t.to(dtype=torch.bfloat16)
            return t

        if self.cfg.common.fp16:
            sample = utils.apply_to_sample(apply_half, sample)

        if self.cfg.common.bf16:
            sample = utils.apply_to_sample(apply_bfloat16, sample)

        if self._dummy_batch == "DUMMY":
            self._dummy_batch = sample

        return sample, False

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate at the end of the epoch."""
        self.lr_scheduler.step(epoch, val_loss)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def get_lr(self):
        """Get the current learning rate."""
        return self.model.optimizer.get_lr()

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def lr_step_update(self):
        """Update the learning rate after each update."""
        new_lr = self.lr_scheduler.step_update(self.get_num_updates())
        if isinstance(new_lr, dict):
            for k, v in new_lr.items():
                metrics.log_scalar(f"lr_{k}", v, weight=0, priority=300)
            new_lr = new_lr.get("default", next(iter(new_lr.values())))
        else:
            metrics.log_scalar("lr", new_lr, weight=0, priority=300)
        return new_lr

    def begin_epoch(self, epoch):
        """Called at the beginning of each epoch."""
        logger.info("begin training epoch {}".format(epoch))

        self.lr_step_begin_epoch(epoch)

        # task specific setup per epoch
        self.task.begin_epoch(epoch, self.get_model())

    def lr_step_begin_epoch(self, epoch):
        """Adjust the learning rate at the beginning of the epoch."""
        self.lr_scheduler.step_begin_epoch(epoch)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model

    def get_criterion(self):
        """Get the (non-wrapped) criterion instance."""
        return self._criterion

    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.cfg.common.seed + self.get_num_updates()
        utils.set_torch_seed(seed)

    def consolidate_optimizer(self):
        """ DeepSpeed doesn't require any optimizer consolidation. """
        return False

    @metrics.aggregate("valid")
    def valid_step(self, sample, raise_oom=False):
        """Do forward pass in evaluation mode."""

        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()

            sample, is_dummy_batch = self._prepare_sample(sample)

            try:
                _loss, sample_size, logging_output = self.task.valid_step(
                    sample, self.model, self.criterion
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if not raise_oom:
                        logger.warning(
                            "ran out of memory in validation step, retrying batch"
                        )
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad = None  # free some memory
                        if self.cuda:
                            torch.cuda.empty_cache()
                        return self.valid_step(sample, raise_oom=True)
                raise e

            logging_outputs = [logging_output]
            if is_dummy_batch:
                if torch.is_tensor(sample_size):
                    sample_size.zero_()
                else:
                    sample_size *= 0.0

        # gather logging outputs from all replicas
        if self.data_parallel_world_size > 1:
            logging_outputs, (sample_size,) = self._aggregate_logging_outputs(
                logging_outputs,
                sample_size,
                ignore=is_dummy_batch,
            )

        # log validation stats
        if self.tpu:
            logging_outputs = self._xla_markstep_and_send_to_cpu(logging_outputs)
        # don't reduce here, otherwise the metric is wrong
        # logging_output = self._reduce_and_log_stats(logging_outputs, sample_size)

        return logging_outputs

    def begin_valid_epoch(self, epoch):
        """Called at the beginning of each validation epoch."""

        # task specific setup per validation epoch
        self.task.begin_valid_epoch(epoch, self.get_model())

    def get_valid_iterator(
        self,
        subset,
        disable_iterator_cache=False,
    ):
        """Return an EpochBatchIterator over given validation subset for a given epoch."""
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.dataset(subset),
            max_tokens=self.cfg.dataset.max_tokens_valid,
            max_sentences=self.cfg.dataset.batch_size_valid,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
            ),
            ignore_invalid_inputs=self.cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple,
            seed=self.cfg.common.seed,
            num_shards=self.data_parallel_world_size,
            shard_id=self.data_parallel_rank,
            num_workers=self.cfg.dataset.num_workers,
            # always pass a fixed "epoch" to keep validation data consistent
            # across training epochs
            epoch=1,
            data_buffer_size=self.cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
        )
        self.reset_dummy_batch(batch_iterator.first_batch)
        return batch_iterator

    def load_checkpoint(
        self,
        filename,
        reset_optimizer=False,
        reset_lr_scheduler=False,
        optimizer_overrides=None,
        reset_meters=False,
    ):
        """
        Load all training state from a checkpoint file.
        rank = 0 will load the checkpoint, and then broadcast it to all
        other ranks.
        """
        extra_state, self._optim_history, last_optim_state = None, [], None

        logger.info(f"Preparing to load checkpoint {filename}")
        is_distributed = self.data_parallel_world_size > 1
        bexists = os.path.isdir(filename) #PathManager.isfile(filename)
        if not bexists:
            logger.info("No existing checkpoint found {}".format(filename))
            return None

        def load_model(src, dst):
            if torch.distributed.get_rank() == 0:
                print(self.cfg.model)
            dst.load_state_dict(src, strict=False, model_cfg=self.cfg.model)

        load_path, client_states = self.model.load_checkpoint(load_dir=filename, load_optimizer_states=not reset_optimizer, model_f=load_model)

        print(f'[{torch.distributed.get_rank()}] ckpt client states={client_states}')

        assert not utils.has_parameters(self.get_criterion()), "criterion w. params not supported yet"
        extra_state = client_states["extra_state"]

        if not reset_optimizer and not reset_lr_scheduler:
            self.lr_scheduler.load_state_dict(client_states["lr_scheduler_state"])

        self.set_num_updates(client_states["num_updates"])

        self.scaler.loss_scale = client_states["loss_scale"]

        if extra_state is not None:
            itr_state = extra_state["train_iterator"]
            epoch =  itr_state.get("epoch", 1)

            if "previous_training_time" in extra_state:
                self._previous_training_time = extra_state["previous_training_time"]
                self._start_time = time.time()

            self.lr_step(epoch)

            if itr_state.get("version", 1) >= 2 and itr_state["iterations_in_epoch"] == 0:
                # reset meters at start of epoch
                reset_meters = True

            if "metrics" in extra_state and not reset_meters:
                metrics.load_state_dict(extra_state["metrics"])

                # reset TimeMeters, since their start times don't make sense anymore
                for meter in metrics.get_meters("default"):
                    if isinstance(meter, meters.TimeMeter):
                        meter.reset()

            logger.info(
                "Loaded checkpoint {} (epoch {} @ {} updates)".format(
                    filename, epoch, self.get_num_updates()
                )
            )
        return extra_state

    def get_train_iterator(
        self,
        epoch,
        combine=True,
        load_dataset=True,
        data_selector=None,
        shard_batch_itr=True,
        disable_iterator_cache=False,
    ):
        """Return an EpochBatchIterator over the training set for a given epoch."""
        if load_dataset:
            logger.info("loading train data for epoch {}".format(epoch))
            self.task.load_dataset(
                self.cfg.dataset.train_subset,
                epoch=epoch,
                combine=combine,
                data_selector=data_selector,
                tpu=self.tpu,
            )
        
        seed = self.cfg.common.seed
        if hasattr(self.cfg, 'infinibatch_dataloader'):
            print("| If using infinibatch, do reset_dataloader!", flush=True)
            assert self.cfg.reset_dataloader
            seed += self.get_num_updates()
            print("| Set seed {}={}+{}".format(seed, self.cfg.common.seed, self.get_num_updates()), flush=True)

        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.dataset(self.cfg.dataset.train_subset),
            max_tokens=self.cfg.dataset.max_tokens,
            max_sentences=self.cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
                self.cfg.dataset.max_tokens,
            ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple,
            seed=seed,
            num_shards=self.data_parallel_world_size if shard_batch_itr else 1,
            shard_id=self.data_parallel_rank if shard_batch_itr else 0,
            num_workers=self.cfg.dataset.num_workers,
            epoch=epoch,
            data_buffer_size=self.cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
        )
        self.reset_dummy_batch(batch_iterator.first_batch)
        return batch_iterator

    def reset_dummy_batch(self, batch):
        self._dummy_batch = batch

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def model(self):
        if self._wrapped_model is None:
            self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def criterion(self):
        if self._wrapped_criterion is None:
            self._wrapped_criterion = self._criterion
        return self._wrapped_criterion

    @property
    def data_parallel_world_size(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 1
        return distributed_utils.get_data_parallel_world_size()

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._build_optimizer()  # this will initialize self._lr_scheduler
        return self._lr_scheduler

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    @property
    def data_parallel_process_group(self):
        return distributed_utils.get_data_parallel_group()

    @property
    def data_parallel_rank(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 0
        return distributed_utils.get_data_parallel_rank()

    @property
    def is_data_parallel_master(self):
        # NOTE: this returns true for all model parallel replicas with data
        # parallel rank 0
        return self.data_parallel_rank == 0

    @property
    def checkpoint_suffix(self) -> str:
        """Suffix to add to the checkpoint file name."""
        # if self.cfg.distributed_training.ddp_backend == "fully_sharded":
        #     return self.cfg.checkpoint.checkpoint_suffix + "-shard{0}".format(self.data_parallel_rank)
        # else:
        return self.cfg.checkpoint.checkpoint_suffix or ""

    @property
    def should_save_checkpoint_on_current_rank(self) -> bool:
        """Indicates whether to save checkpoints on the current DDP rank."""
        if self.zero_enabled:
            return True
        else:
            return self.is_data_parallel_master

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        logger.info(f"Saving checkpoint to {filename}")
        
        state_dict = self.state_dict(exclude_model_opt=True)
        state_dict["extra_state"].update(extra_state)

        self.model.save_checkpoint(save_dir=filename, client_state=state_dict)

        logger.info(f"Finished saving checkpoint to {filename}")

    def state_dict(self, exclude_model_opt=False):
        state_dict = {
            "args": None,  # legacy
            "cfg": (
                OmegaConf.to_container(self.cfg)
                if OmegaConf.is_config(self.cfg) else self.cfg
            ),
            "model": None if exclude_model_opt else self.model.state_dict(),
            "criterion": (
                self.criterion.state_dict()
                if utils.has_parameters(self.criterion) else None
            ),
            "optimizer_history": (self._optim_history or [])
            + [
                {
                    "criterion_name": self.get_criterion().__class__.__name__,
                    "optimizer_name": self.optimizer.__class__.__name__,
                    "lr_scheduler_state": self.lr_scheduler.state_dict(),
                    "num_updates": self.get_num_updates(),
                }
            ],
            "lr_scheduler_state": self.lr_scheduler.state_dict(),
            "num_updates": self.get_num_updates(),
            "task_state": self.task.state_dict() if self.task is not None else {},
            "loss_scale": self.scaler.loss_scale,
            "extra_state": {
                "metrics": metrics.state_dict(),
                "previous_training_time": self.cumulative_training_time(),
            }
        }
        if exclude_model_opt and not self.cfg.checkpoint.no_save_optimizer_state:
            state_dict["last_optimizer_state"] = self.optimizer.state_dict()
        return state_dict

def _catalog_shared_params(module, memo=None, prefix=""):
    if memo is None:
        first_call = True
        memo = {}
    else:
        first_call = False
    for name, param in module._parameters.items():
        param_prefix = prefix + ("." if prefix else "") + name
        if param not in memo:
            memo[param] = []
        memo[param].append(param_prefix)
    for name, m in module._modules.items():
        if m is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        _catalog_shared_params(m, memo, submodule_prefix)
    if first_call:
        return [x for x in memo.values() if len(x) > 1]


def _get_module_by_path(module, path):
    path = path.split(".")
    for name in path:
        module = getattr(module, name)
    return module


def _set_module_by_path(module, path, value):
    path = path.split(".")
    for name in path[:-1]:
        module = getattr(module, name)
    setattr(module, path[-1], value)
