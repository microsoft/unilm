# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, List, Optional, Sequence, Union, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.nn import Module as _DINOModel
from torchmetrics import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TRANSFORMERS_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchvision import transforms
from torchvision.transforms import Compose as _DINOProcessor
from typing_extensions import Literal

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["DINOScore.plot"]

_DEFAULT_MODEL: str = "dino_vits16"


class DINOScore(Metric):
    r"""Calculates `DINO Score`_ which is a image-to-image similarity metric.

    .. note:: Metric is not scriptable

    Args:
        model_name_or_path: string indicating the version of the DINO model to use. Available models are:

            - `"dino_vits16"`

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If transformers package is not installed or version is lower than 4.10.0
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0

    score: Tensor
    n_samples: Tensor
    plot_upper_bound = 100.0

    def __init__(
            self,
            model_name_or_path: Literal[
                "dino_vits16",
            ] = _DEFAULT_MODEL,  # type: ignore[assignment]
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model, self.processor = self._get_model_and_processor(model_name_or_path)
        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    @staticmethod
    def _get_model_and_processor(
            model_name_or_path: Literal[
                "dino_vits16",
            ] = "dino_vits16",
    ) -> Tuple[_DINOModel, _DINOProcessor]:
        if _TRANSFORMERS_AVAILABLE:
            model = torch.hub.load('facebookresearch/dino:main', model_name_or_path)
            processor = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            return model, processor

        raise ModuleNotFoundError(
            "`dino_score` metric requires `transformers` package be installed."
            " Either install with `pip install transformers>=4.0` or `pip install torchmetrics[multimodal]`."
        )

    @staticmethod
    def _dino_score_update(
            images1: Union[Image.Image, List[Image.Image]],
            images2: Union[Image.Image, List[Image.Image]],
            model: _DINOModel,
            processor: _DINOProcessor,
    ) -> Tuple[Tensor, int]:
        if len(images1) != len(images2):
            raise ValueError(
                f"Expected the number of images to be the same but got {len(images1)} and {len(images2)}"
            )

        device = next(model.parameters()).device

        img1_processed_input = [processor(i) for i in images1]
        img2_processed_input = [processor(i) for i in images2]

        img1_processed_input = torch.stack(img1_processed_input).to(device)
        img2_processed_input = torch.stack(img2_processed_input).to(device)

        img1_features = model(img1_processed_input)
        img2_features = model(img2_processed_input)

        # cosine similarity between feature vectors
        score = 100 * F.cosine_similarity(img1_features, img2_features, dim=-1)
        return score, len(images1)

    def update(self, images1: Union[Image.Image, List[Image.Image]],
               images2: Union[Image.Image, List[Image.Image]]) -> None:
        """Update DINO score on a batch of images and text.

        Args:
            images1: Either a single [N, C, H, W] tensor or a list of [C, H, W] tensors
            images2: Either a single [N, C, H, W] tensor or a list of [C, H, W] tensors

        Raises:
            ValueError:
                If not all images have format [C, H, W]
            ValueError:
                If the number of images do not match
        """
        score, n_samples = self._dino_score_update(images1, images2, self.model, self.processor)
        self.score += score.sum(0)
        self.n_samples += n_samples

    def compute(self) -> Tensor:
        """Compute accumulated dino score."""
        return torch.max(self.score / self.n_samples, torch.zeros_like(self.score))

    def plot(self, val: Union[Tensor, Sequence[Tensor], None] = None, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed
        """
        return self._plot(val, ax)
