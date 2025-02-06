import torch.distributed as dist
from typing import List, Any, Dict
import torch
import numpy as np


class DistGatherMixin:
    def gather(self):
        pass

    @staticmethod
    def gather_object(objects: List[Any]):
        output = [None for _ in range(dist.get_world_size())]
        dist.gather_object(objects,
                           object_gather_list=output if dist.get_rank() == 0 else None,
                           dst=0)

        if dist.get_rank() == 0:
            return output
        else:
            return None


class SFTLossOnlyPostProcessor(DistGatherMixin):
    def __init__(self):
        super().__init__()
        self.losses = []

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        loss = batch_model_outputs["loss"].item()

        if ddp:
            gather_res = self.gather_object(loss)
            if dist.get_rank() == 0:
                loss = sum(gather_res) / len(gather_res)

        self.losses.append(loss)

    def get_results(self, output_dir: str):
        avg_loss = np.mean(self.losses).item()

        metrics = {
            "loss": avg_loss,
        }

        return metrics, []
