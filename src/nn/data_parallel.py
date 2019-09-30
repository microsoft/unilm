import torch
from torch.nn import DataParallel
from torch.cuda._utils import _get_device_index
from torch.nn.parallel._functions import Scatter
from itertools import chain


def scatter_imbalance(inputs, target_gpus, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            if (len(target_gpus) == 4) and (obj.size(dim) == 22):
                return Scatter.apply(target_gpus, (4, 6, 6, 6), dim, obj)
            if (len(target_gpus) == 4) and (obj.size(dim) == 60):
                return Scatter.apply(target_gpus, (12, 16, 16, 16), dim, obj)
            elif (len(target_gpus) == 4) and (obj.size(dim) == 144):
                return Scatter.apply(target_gpus, (24, 40, 40, 40), dim, obj)
            elif (len(target_gpus) == 8) and (obj.size(dim) == 46):
                return Scatter.apply(target_gpus, (4, 6, 6, 6, 6, 6, 6, 6), dim, obj)
            elif (len(target_gpus) == 8) and (obj.size(dim) == 62):
                return Scatter.apply(target_gpus, (6, 8, 8, 8, 8, 8, 8, 8), dim, obj)
            elif (len(target_gpus) == 8) and (obj.size(dim) == 94):
                return Scatter.apply(target_gpus, (10, 12, 12, 12, 12, 12, 12, 12), dim, obj)
            elif (len(target_gpus) == 8) and (obj.size(dim) == 110):
                return Scatter.apply(target_gpus, (12, 14, 14, 14, 14, 14, 14, 14), dim, obj)
            elif (len(target_gpus) == 8) and (obj.size(dim) == 118):
                return Scatter.apply(target_gpus, (13, 15, 15, 15, 15, 15, 15, 15), dim, obj)
            elif (len(target_gpus) == 8) and (obj.size(dim) == 126):
                return Scatter.apply(target_gpus, (14, 16, 16, 16, 16, 16, 16, 16), dim, obj)
            elif (len(target_gpus) == 8) and (obj.size(dim) == 134):
                return Scatter.apply(target_gpus, (15, 17, 17, 17, 17, 17, 17, 17), dim, obj)
            elif (len(target_gpus) == 8) and (obj.size(dim) == 142):
                return Scatter.apply(target_gpus, (16, 18, 18, 18, 18, 18, 18, 18), dim, obj)
            elif (len(target_gpus) == 16) and (obj.size(dim) == 222):
                return Scatter.apply(target_gpus, (12, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14), dim, obj)
            return Scatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs_imbalance(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter_imbalance(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter_imbalance(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class DataParallelImbalance(DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallelImbalance, self).__init__(
            module, device_ids, output_device, dim)

        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]

        if not all(t.is_cuda and t.device.index == device_ids[0]
                   for t in chain(module.parameters(), module.buffers())):
            raise RuntimeError("module must have its parameters and buffers "
                               "on device %d (device_ids[0])" % device_ids[0])

        self.dim = dim
        self.module = module
        self.device_ids = list(
            map(lambda x: _get_device_index(x, True), device_ids))
        self.output_device = _get_device_index(output_device, True)

        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter_imbalance(
            inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def scatter_imbalance(self, inputs, kwargs, device_ids):
        return scatter_kwargs_imbalance(inputs, kwargs, device_ids, dim=self.dim)
