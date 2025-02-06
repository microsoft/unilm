import os
from typing import Optional, Union

import torch
from fairscale.nn.model_parallel import initialize as mpu
from transformers import PreTrainedModel

from models.mixin import PreTrainedModelPeftMixin, return_reference_model, set_reference_model


class PretrainedModelParallelPreSplitMixin(PreTrainedModelPeftMixin):
    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
            *model_args,
            **kwargs,
    ):
        if mpu.model_parallel_is_initialized():
            mp_rank = mpu.get_model_parallel_rank()
            pretrained_model_name_or_path = os.path.join(pretrained_model_name_or_path, f"mp_{mp_rank}-of-{mpu.get_model_parallel_world_size()}")
            print(f"Loading model from {pretrained_model_name_or_path}")
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            *args,
            **kwargs,
    ):
        if mpu.model_parallel_is_initialized():
            mp_rank = mpu.get_model_parallel_rank()
            save_directory = os.path.join(save_directory, f"mp_{mp_rank}-of-{mpu.get_model_parallel_world_size()}")
        super().save_pretrained(save_directory, *args, **kwargs)

    @classmethod
    def from_pretrained_with_ref_model(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], ref_model: PreTrainedModel,
                                       *model_args, **kwargs):
        set_reference_model(ref_model)
        ref_model = return_reference_model()
        ref_model.eval()
        ref_model.to(device=torch.cuda.current_device())

        model = cls.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model
