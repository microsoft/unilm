import functools

import torch
from fairscale.nn.data_parallel.fully_sharded_data_parallel import FullyShardedDataParallel as FullyShardedDP
from fairscale.nn.wrap.auto_wrap import auto_wrap, enable_wrap, default_auto_wrap_policy

from general_util.logger import get_child_logger
from torch import nn

logger = get_child_logger("FSDPUtils")


def default_initialize(model: torch.nn.Module,
                       device: torch.device,
                       fp16: bool = False,
                       flatten_parameters: bool = True,
                       disable_reshard_on_root: bool = True,
                       reshard_after_forward: bool = True,
                       move_grads_to_cpu: bool = False,
                       move_params_to_cpu: bool = False):
    fsdp_params = dict(mixed_precision=fp16,
                       flatten_parameters=flatten_parameters,
                       disable_reshard_on_root=disable_reshard_on_root,
                       reshard_after_forward=reshard_after_forward,
                       move_grads_to_cpu=move_grads_to_cpu,
                       move_params_to_cpu=move_params_to_cpu)

    # Better speed

    logger.info(fsdp_params)

    model = FullyShardedDP(model, **fsdp_params)

    if not move_params_to_cpu:
        model = model.to(device)

    logger.info(model)

    return model


def vae_specific_initialize(model: torch.nn.Module,
                            device: torch.device,
                            fp16: bool = False,
                            flatten_parameters: bool = True,
                            disable_reshard_on_root: bool = True,
                            reshard_after_forward: bool = True,
                            move_grads_to_cpu: bool = False,
                            move_params_to_cpu: bool = False,
                            min_num_params: int = 1e8):
    from transformers.models.bart.modeling_bart import BartDecoderLayer, BartDecoder

    # Better memory?
    wrap_policy = functools.partial(default_auto_wrap_policy,
                                    module_is_root=True,
                                    # force_leaf_modules=force_leaf_modules,
                                    min_num_params=min_num_params,
                                    exclude_wrap_modules={nn.ModuleList, nn.ModuleDict})
    fsdp_params = dict(mixed_precision=fp16,
                       flatten_parameters=flatten_parameters,
                       disable_reshard_on_root=disable_reshard_on_root,
                       reshard_after_forward=reshard_after_forward,
                       move_grads_to_cpu=move_grads_to_cpu,
                       move_params_to_cpu=move_params_to_cpu)
    with enable_wrap(wrapper_cls=FullyShardedDP, auto_wrap_policy=wrap_policy, **fsdp_params):
        model = auto_wrap(model)
        model = FullyShardedDP(model, **fsdp_params)

    logger.info(model)

    assert isinstance(model, FullyShardedDP)

    if not move_params_to_cpu:
        model = model.to(device)

    return model


def recursive_initialize(model: torch.nn.Module,
                         device: torch.device,
                         fp16: bool = False,
                         flatten_parameters: bool = True,
                         disable_reshard_on_root: bool = True,
                         reshard_after_forward: bool = True,
                         move_grads_to_cpu: bool = False,
                         move_params_to_cpu: bool = False,
                         min_num_params: int = 1e8):
    # Better memory?
    wrap_policy = functools.partial(default_auto_wrap_policy,
                                    module_is_root=True,
                                    # force_leaf_modules=force_leaf_modules,
                                    min_num_params=min_num_params)
    fsdp_params = dict(mixed_precision=fp16,
                       flatten_parameters=flatten_parameters,
                       disable_reshard_on_root=disable_reshard_on_root,
                       reshard_after_forward=reshard_after_forward,
                       move_grads_to_cpu=move_grads_to_cpu,
                       move_params_to_cpu=move_params_to_cpu)
    with enable_wrap(wrapper_cls=FullyShardedDP, auto_wrap_policy=wrap_policy, **fsdp_params):
        model = auto_wrap(model)
    model = FullyShardedDP(model, **fsdp_params)

    logger.info(model)

    assert isinstance(model, FullyShardedDP)

    if not move_params_to_cpu:
        model = model.to(device)

    return model


def default_initialize_v2(model: torch.nn.Module,
                          device: torch.device,
                          fp16: bool = False,
                          flatten_parameters: bool = True,
                          disable_reshard_on_root: bool = True,
                          reshard_after_forward: bool = True,
                          move_grads_to_cpu: bool = False,
                          move_params_to_cpu: bool = False,
                          min_num_params: int = 1e8):
    # Better memory?
    wrap_policy = functools.partial(default_auto_wrap_policy,
                                    module_is_root=True,
                                    # force_leaf_modules=force_leaf_modules,
                                    exclude_wrap_modules={nn.ModuleDict},
                                    min_num_params=min_num_params)
    fsdp_params = dict(mixed_precision=fp16,
                       flatten_parameters=flatten_parameters,
                       disable_reshard_on_root=disable_reshard_on_root,
                       reshard_after_forward=reshard_after_forward,
                       move_grads_to_cpu=move_grads_to_cpu,
                       move_params_to_cpu=move_params_to_cpu)
    with enable_wrap(wrapper_cls=FullyShardedDP, auto_wrap_policy=wrap_policy, **fsdp_params):
        model = auto_wrap(model)
    model = FullyShardedDP(model, **fsdp_params)

    logger.info(model)

    assert isinstance(model, FullyShardedDP)

    if not move_params_to_cpu:
        model = model.to(device)

    return model
