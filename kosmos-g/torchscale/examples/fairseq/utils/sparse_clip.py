# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math
import warnings

import torch
import torch.distributed as dist
from fairseq.utils import multi_tensor_l2norm_available, multi_tensor_total_norm


@torch.no_grad()
def clip_grad_norm_(
    params, max_norm, moe_expert_count, aggregate_norm_fn=None
) -> torch.Tensor:
    def grad_exists(p):
        return p is not None and getattr(p, "grad", None) is not None

    if isinstance(params, torch.Tensor):
        params = [params]
    params = list(params)
    params = list(filter(grad_exists, params))
    grads, expert_grads, base_expert_grads, sharded_grads = [], [], [], []
    denom = math.sqrt(max(dist.get_global_world_size(), moe_expert_count))
    for p in params:
        if hasattr(p, "expert"):
            expert_grads.append(p.grad.detach() / denom)
        elif hasattr(p, "base_expert"):
            base_expert_grads.append(p.grad.detach())
        elif hasattr(p, "_is_sharded"):
            sharded_grads.append(p.grad.detach())
        else:
            grads.append(p.grad.detach())
    if len(grads) == 0:
        if len(params) > 0:
            total_norm = params[0].new_tensor(0.0)
        else:
            total_norm = torch.tensor(0.0)
    elif len(grads) == 1:
        total_norm = torch.norm(grads[0], p=2, dtype=torch.float32)
    else:
        if multi_tensor_l2norm_available:
            total_norm = multi_tensor_total_norm(grads)
        else:
            if torch.cuda.is_available():
                warnings.warn(
                    "amp_C fused kernels unavailable, disabling multi_tensor_l2norm; "
                    "you may get better performance by installing NVIDIA's apex library"
                )
                device = torch.cuda.current_device()
            elif grads[0].device.type == "xla":
                device = grads[0].device
            else:
                device = torch.device("cpu")
            total_norm = torch.norm(
                torch.stack(
                    [torch.norm(g, p=2, dtype=torch.float32).to(device) for g in grads]
                )
            )

    # calculate split_norm and all_reduce with other workers
    norms = [total_norm]
    for split_grads in [expert_grads, sharded_grads]:
        if len(split_grads) == 0:
            continue
        split_norm = torch.norm(
            torch.stack([torch.norm(g, p=2, dtype=torch.float32) for g in split_grads])
        )
        if dist.is_initialized():
            split_norm.pow_(2)
            dist.all_reduce(split_norm)
            split_norm.sqrt_()
        norms.append(split_norm)
    if len(norms) > 1:
        total_norm = torch.norm(torch.stack(norms))

    if aggregate_norm_fn is not None:
        total_norm = aggregate_norm_fn(total_norm)

    if max_norm > 0:
        max_norm = float(max_norm)
        clip_coef = (max_norm / (total_norm + 1e-6)).clamp_(max=1)
        for g in grads + expert_grads + sharded_grads + base_expert_grads:
            g.mul_(clip_coef)
    return total_norm
