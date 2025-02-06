from transformers import PreTrainedModel
import deepspeed
from fairscale.nn.model_parallel import initialize as mpu
from omegaconf import DictConfig, OmegaConf
from general_util import training_utils


def init_ds_training_engine(model: PreTrainedModel, ds_cfg: DictConfig, global_cfg: DictConfig, ):
    ds_config = ds_cfg
    if "total_num_steps" in ds_config.scheduler.params:
        ds_config.scheduler.params.total_num_steps = global_cfg.max_steps
    ds_config.scheduler.params.warmup_num_steps = global_cfg.warmup_steps
    ds_config = OmegaConf.to_container(ds_config, resolve=True)
    ds_config["train_mirco_batch_size_per_gpu"] = global_cfg.per_gpu_train_batch_size

    optim_params = training_utils.get_optimizer_grouped_parameters(model, global_cfg.actor_weight_decay)

    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=optim_params,
        config_params=ds_config,
        mpu=mpu if mpu.model_parallel_is_initialized() else None,
    )

    return engine, optimizer, scheduler


def init_ds_eval_engine(model: PreTrainedModel, ds_cfg: DictConfig):
    ds_config = ds_cfg
    if ds_config.zero_optimization.stage != 3:
        ds_config.zero_optimization.stage = 0

    ds_config = OmegaConf.to_container(ds_config, resolve=True)
    # ds_config["train_mirco_batch_size_per_gpu"] = global_cfg.per_gpu_train_batch_size
    if "optimizer" in ds_config:
        ds_config.pop("optimizer")
    if "scheduler" in ds_config:
        ds_config.pop("scheduler")

    engine, *_ = deepspeed.initialize(
        model=model,
        config_params=ds_config,
        mpu=mpu if mpu.model_parallel_is_initialized() else None,
    )

    return engine
