from megatron.core import parallel_state
import torch
from torch.utils.data.distributed import DistributedSampler

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


def get_model_parallel_group():
    return parallel_state.get_tensor_model_parallel_group()


def get_model_parallel_rank():
    return parallel_state.get_tensor_model_parallel_rank()


def get_model_parallel_world_size():
    return parallel_state.get_tensor_model_parallel_world_size()


def get_data_parallel_group():
    return parallel_state.get_data_parallel_group()


def get_data_parallel_rank():
    return parallel_state.get_data_parallel_rank()


def get_data_parallel_world_size():
    return parallel_state.get_data_parallel_world_size()


def prepare_distributed_sampler(dataset: torch.utils.data.Dataset, random_seed: int = 42, shuffle: bool = True):
    if parallel_state.model_parallel_is_initialized():
        sub_train_sampler = DistributedSampler(dataset,
                                               shuffle=shuffle,
                                               num_replicas=parallel_state.get_data_parallel_world_size(),
                                               rank=parallel_state.get_data_parallel_rank(),
                                               seed=random_seed)
    else:
        sub_train_sampler = DistributedSampler(dataset, shuffle=shuffle)

    logger.info(f"Distributed Shuffling: {shuffle}")
    return sub_train_sampler
