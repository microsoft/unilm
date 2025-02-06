from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from general_util.average_meter import LogMetric, AverageMeter
from general_util.logger import get_child_logger

logger = get_child_logger("Mixin")


class LogMixin:
    eval_metrics: LogMetric = None

    def init_metric(self, *metric_names):
        self.eval_metrics = LogMetric(*metric_names)

    def get_eval_log(self, reset=False, ddp=False, device='cpu'):

        if self.eval_metrics is None:
            logger.warning("The `eval_metrics` attribute hasn't been initialized.")

        if ddp:
            for metric in self.eval_metrics.metrics.values():
                metric.gather(device=device)

        results = self.eval_metrics.get_log()

        _eval_metric_log = '\t'.join([f"{k}: {v}" for k, v in results.items()])

        if reset:
            self.eval_metrics.reset()

        return _eval_metric_log, results


class MetricMixin:
    # TODO: 如何利用hydra解耦计算metric的方式和模型？
    def __init__(self, metrics: List[Tuple[str, str, str, str]]):
        self.metrics = {
            name: {
                "key": key,
                "val": val,
                "func": func,
                "meter": AverageMeter()
            } for key, val, func, name in metrics
        }


class PredictionMixin:
    tensor_dict: Dict[str, List] = defaultdict(list)

    def reset_predict_tensors(self):
        self.tensor_dict = defaultdict(list)

    def concat_predict_tensors(self, **tensors: torch.Tensor):
        for k, v in tensors.items():
            self.tensor_dict[k].extend(v.detach().cpu().tolist())

    def get_predict_tensors(self):
        return self.tensor_dict
