import glob
import os
import random
import re
from typing import Dict, List

import hydra
import numpy as np
import omegaconf
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from general_util.logger import get_child_logger

logger = get_child_logger("TrainingUtils")


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def set_seed_int(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return -1


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).
    Args:
        model (:obj:`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


def get_zero_stage(cfg: DictConfig):
    if hasattr(cfg, "zero_optimization"):
        return int(getattr(cfg.zero_optimization, "stage", 0))
    return 0


def return_torch_dtype(dtype: str):
    if dtype == "float16":
        return torch.float16
    elif dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "float32":
        return torch.float32
    elif dtype == "int8":
        return torch.int8
    else:
        return dtype


def batch_to_device(batch: Dict[str, torch.Tensor], device):
    if "meta_data" in batch:
        batch.pop("meta_data")
    if "index" in batch:
        batch.pop("index")

    batch_on_device = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_on_device[k] = v.to(device)
        else:
            batch_on_device[k] = v
    return batch_on_device


def initialize_dataset(cfg: DictConfig, file_path: str, tokenizer: PreTrainedTokenizer):
    if "_target_" in cfg:
        return hydra.utils.call(cfg, file_path=file_path, tokenizer=tokenizer)
    else:
        datasets = [initialize_dataset(cfg[key], file_path, tokenizer) for key in cfg.keys()]
        assert len(datasets)
        datasets = ConcatDataset(datasets)
        return datasets


def load_and_cache_examples(cfg, tokenizer: PreTrainedTokenizer, _split="train", _file: str = None):
    if_barrier = False

    if _file is not None:
        input_file = _file
        if_barrier = True
    else:
        if _split == "train":
            input_file = cfg.train_file
            if_barrier = True
        elif _split == "dev":
            input_file = cfg.dev_file
            if cfg.ddp_eval and cfg.local_rank != -1:
                if_barrier = True
        elif _split == "test":
            input_file = cfg.test_file
            if cfg.ddp_eval and cfg.local_rank != -1:
                if_barrier = True
        else:
            raise RuntimeError(_split)

    if getattr(cfg, "dist_load_data_barrier", True) and if_barrier and cfg.local_rank not in [-1, 0]:
        dist.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    logger.info("Creating features from dataset file at %s", input_file)
    sub_config = f"read_tensor_{_split}"
    if sub_config in cfg:
        dataset = initialize_dataset(cfg[sub_config], file_path=input_file, tokenizer=tokenizer)
    else:
        dataset = initialize_dataset(cfg.read_tensor, file_path=input_file, tokenizer=tokenizer)

    if getattr(cfg, "dist_load_data_barrier", True) and if_barrier and cfg.local_rank == 0:
        dist.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    return dataset


def organize_multiple_dataset(cfg, tokenizer: PreTrainedTokenizer, _split="train"):
    if "_target_" in cfg.train_file:
        files = hydra.utils.instantiate(cfg.train_file)
    elif isinstance(cfg.train_file, omegaconf.ListConfig):
        files = list(cfg.train_file)
    elif cfg.train_file.startswith("hf:"):
        files = [cfg.train_file[3:]]
    elif cfg.train_file.startswith("list:"):
        files = [cfg.train_file[5:]]
    elif os.path.exists(cfg.train_file):
        files = [cfg.train_file]
    else:
        files = list(glob.glob(cfg.train_file))
    logger.info(files)

    if getattr(cfg, "total_dataset_len", -1) > 0:
        total_dataset_len = cfg.total_dataset_len
    else:
        total_dataset_len = 0
        if dist.is_initialized() and dist.get_rank() != 0:
            dist.barrier()

        if not dist.is_initialized() or dist.get_rank() == 0:
            for _file in tqdm(files, total=len(files)):
                sub_train_dataset = load_and_cache_examples(cfg, tokenizer, _split="train", _file=_file)
                total_dataset_len += len(sub_train_dataset)
                del sub_train_dataset

            if dist.is_initialized():
                dist.barrier()

        if dist.is_initialized():
            if dist.get_rank() == 0:
                objects = [total_dataset_len for _ in range(dist.get_world_size())]
            else:
                objects = [None for _ in range(dist.get_world_size())]
            output_list = [None]
            dist.scatter_object_list(output_list, objects, src=0)
            if dist.get_rank() != 0:
                total_dataset_len = output_list[0]

            assert total_dataset_len > 0

    logger.warning(f"Rank No. {cfg.local_rank} has {total_dataset_len} samples.")
    cfg.total_dataset_len = total_dataset_len

    return files, total_dataset_len


def if_cancel_sync(cfg: DictConfig, step: int):
    if getattr(cfg, "forward_sync", False) is False and (
            step + 1) % cfg.gradient_accumulation_steps != 0 and cfg.local_rank != -1:
        return True
    return False


def initialize_optimizer(cfg: DictConfig, grouped_parameters: List[Dict] = None, model: torch.nn.Module = None):
    if grouped_parameters is None:
        assert model is not None, "Either ``grouped_parameters`` or ``model`` must be specified."
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if
                           (not any(nd in n for nd in no_decay)) and p.requires_grad],
                'weight_decay': cfg.weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() if
                           (any(nd in n for nd in no_decay)) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]

    if "optimizer" in cfg and cfg.optimizer and 'lamb' in cfg.optimizer:
        if "bit_training" in cfg and cfg.bit_training:
            from bitsandbytes.optim import LAMB8bit

            optimizer = LAMB8bit(grouped_parameters,
                                 lr=cfg.learning_rate,
                                 betas=eval(cfg.adam_betas),
                                 eps=cfg.adam_epsilon,
                                 max_unorm=cfg.max_grad_norm)
        else:
            if cfg.optimizer == 'fused_lamb':
                try:
                    from apex.optimizers.fused_mixed_precision_lamb import FusedMixedPrecisionLamb as FusedLAMB
                except ImportError:
                    from apex.optimizers.fused_lamb import FusedLAMB
            else:
                from apex.optimizers.fused_lamb import FusedLAMB

            optimizer = FusedLAMB(grouped_parameters,
                                  lr=cfg.learning_rate,
                                  betas=eval(cfg.adam_betas),
                                  eps=cfg.adam_epsilon,
                                  use_nvlamb=(cfg.use_nvlamb if "use_nvlamb" in cfg else False),
                                  max_grad_norm=cfg.max_grad_norm)
    elif "optimizer" in cfg and cfg.optimizer and "adafactor" in cfg.optimizer:
        from transformers.optimization import Adafactor

        optimizer = Adafactor(
            grouped_parameters,
            lr=cfg.learning_rate,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )
    else:
        if "bit_training" in cfg and cfg.bit_training:
            from bitsandbytes.optim import AdamW8bit

            optimizer = AdamW8bit(grouped_parameters, lr=cfg.learning_rate, eps=cfg.adam_epsilon,
                                  betas=(eval(cfg.adam_betas)))
        else:
            if hasattr(cfg, "multi_tensor") and cfg.multi_tensor:
                from torch.optim._multi_tensor import AdamW
            else:
                from torch.optim.adamw import AdamW

            optimizer = AdamW(grouped_parameters, lr=cfg.learning_rate, eps=cfg.adam_epsilon,
                              betas=(eval(cfg.adam_betas)))

    return optimizer


def get_optimizer_grouped_parameters(
        model,
        weight_decay,
        lora_lr=5e-4,
        no_decay_name_list=("bias", "LayerNorm.weight", "layernorm.weight"),
        # lora_name_list=("lora_right_weight", "lora_left_weight"),
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                        not any(nd in n for nd in no_decay_name_list)
                        and p.requires_grad
                    # and not any(nd in n for nd in lora_name_list)
                )
            ],
            "weight_decay": weight_decay,
        },
        # {
        #     "params": [
        #         p
        #         for n, p in model.named_parameters()
        #         if (
        #                 not any(nd in n for nd in no_decay_name_list)
        #                 and p.requires_grad
        #                 and any(nd in n for nd in lora_name_list)
        #         )
        #     ],
        #     "weight_decay": weight_decay,
        #     "lr": lora_lr,
        # },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    if not optimizer_grouped_parameters[1]["params"]:
        optimizer_grouped_parameters.pop(1)
    return optimizer_grouped_parameters


def initialize_lr_scheduler(cfg: DictConfig, optimizer, num_warmup_steps: int, num_training_steps: int):
    if hasattr(cfg, "lr_scheduler"):
        if cfg.lr_scheduler == "linear":
            from transformers import get_linear_schedule_with_warmup

            lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        elif cfg.lr_scheduler == "cosine":
            from transformers import get_cosine_schedule_with_warmup

            lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        elif cfg.lr_scheduler == "constant":
            from transformers import get_constant_schedule_with_warmup

            lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
        elif cfg.lr_scheduler == "poly":
            from transformers import get_polynomial_decay_schedule_with_warmup

            lr_scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        else:
            raise NotImplementedError()
    else:
        from transformers import get_linear_schedule_with_warmup

        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    return lr_scheduler


def note_best_checkpoint(cfg: DictConfig, results: Dict[str, float], sub_path: str):
    metric = results[cfg.prediction_cfg.metric]
    if (not cfg.prediction_cfg.best_result) or (
            cfg.prediction_cfg.measure > 0 and metric > cfg.prediction_cfg.best_result) or (
            cfg.prediction_cfg.measure < 0 and metric < cfg.prediction_cfg.best_result):
        cfg.prediction_cfg.best_result = metric
        cfg.prediction_cfg.best_checkpoint = sub_path
        return True
    return False


PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"-(\d+)$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return None
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))
