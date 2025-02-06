from typing import Dict

import torch
import torch.distributed as dist


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        if isinstance(n, torch.Tensor):
            n = n.item()

        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
        else:
            self.avg = 0

    def save(self):
        return {
            'val': self.val,
            'avg': self.avg,
            'sum': self.sum,
            'count': self.count
        }

    def load(self, value: dict):
        if value is None:
            self.reset()
        self.val = value['val'] if 'val' in value else 0
        self.avg = value['avg'] if 'avg' in value else 0
        self.sum = value['sum'] if 'sum' in value else 0
        self.count = value['count'] if 'count' in value else 0

    def gather(self, device):
        tensor_list = [torch.zeros(2, device=device, dtype=torch.float32) for _ in range(dist.get_world_size())]
        tensor = torch.tensor([self.sum, self.count], device=device, dtype=torch.float32)
        dist.all_gather(tensor_list, tensor)

        all_tensor = torch.stack(tensor_list, dim=0)
        self.sum = all_tensor[:, 0].sum().item()
        self.count = all_tensor[:, 1].sum().item()
        if self.count > 0:
            self.avg = self.sum / self.count
        else:
            self.avg = 0

        del all_tensor


class LogMetric(object):
    """
    Record all metrics for logging.
    """

    def __init__(self, *metric_names):

        self.metrics: Dict[str, AverageMeter] = {
            key: AverageMeter() for key in metric_names
        }

    def update(self, metric_name, val, n=1):

        self.metrics[metric_name].update(val, n)

    def reset(self, metric_name=None):
        if metric_name is None:
            for key in self.metrics.keys():
                self.metrics[key].reset()
            return

        self.metrics[metric_name].reset()

    def get_log(self):

        log = {
            key: self.metrics[key].avg for key in self.metrics
        }
        return log
