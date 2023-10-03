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
from PIL import Image
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.multimodal.clip_score import _get_model_and_processor
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TRANSFORMERS_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from typing_extensions import Literal

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["CLIPScore.plot"]

_DEFAULT_MODEL: str = "openai/clip-vit-large-patch14"

if _TRANSFORMERS_AVAILABLE:
    from transformers import CLIPModel as _CLIPModel
    from transformers import CLIPProcessor as _CLIPProcessor


    def _download_clip() -> None:
        _CLIPModel.from_pretrained(_DEFAULT_MODEL)
        _CLIPProcessor.from_pretrained(_DEFAULT_MODEL)


    if _SKIP_SLOW_DOCTEST and not _try_proceed_with_timeout(_download_clip):
        __doctest_skip__ = ["CLIPScore", "CLIPScore.plot"]
else:
    __doctest_skip__ = ["CLIPScore", "CLIPScore.plot"]


class CLIPIScore(Metric):
    r"""Calculates `CLIP Score`_ which is a text-to-image similarity metric.

    CLIP is a reference free metric that can be used to evaluate the correlation between a generated caption for an
    image and the actual content of the image. It has been found to be highly correlated with human judgement. The
    metric is defined as:

    .. math::
        \text{CLIPScore(I, C)} = max(100 * cos(E_I, E_C), 0)

    which corresponds to the cosine similarity between visual CLIP embedding :math:`E_i` for an image :math:`i` and
    textual CLIP embedding :math:`E_C` for an caption :math:`C`. The score is bound between 0 and 100 and the closer
    to 100 the better.

    .. note:: Metric is not scriptable

    Args:
        model_name_or_path: string indicating the version of the CLIP model to use. Available models are:

            - `"openai/clip-vit-base-patch16"`
            - `"openai/clip-vit-base-patch32"`
            - `"openai/clip-vit-large-patch14-336"`
            - `"openai/clip-vit-large-patch14"`

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If transformers package is not installed or version is lower than 4.10.0

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> from torchmetrics.multimodal import CLIPScore
        >>> metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
        >>> score = metric(torch.randint(255, (3, 224, 224)), "a photo of a cat")
        >>> print(score.detach())
        tensor(24.7691)
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
                "openai/clip-vit-base-patch16",
                "openai/clip-vit-base-patch32",
                "openai/clip-vit-large-patch14-336",
                "openai/clip-vit-large-patch14",
            ] = _DEFAULT_MODEL,  # type: ignore[assignment]
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model, self.processor = _get_model_and_processor(model_name_or_path)
        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    @staticmethod
    def _clip_score_update(
            images1: Union[Image.Image, List[Image.Image]],
            images2: Union[Image.Image, List[Image.Image]],
            model: _CLIPModel,
            processor: _CLIPProcessor,
    ) -> Tuple[Tensor, int]:
        if len(images1) != len(images2):
            raise ValueError(
                f"Expected the number of images to be the same but got {len(images1)} and {len(images2)}"
            )

        device = next(model.parameters()).device
        img1_processed_input = processor(images=images1, return_tensors="pt")
        img2_processed_input = processor(images=images2, return_tensors="pt")

        img1_features = model.get_image_features(img1_processed_input["pixel_values"].to(device))
        img1_features = img1_features / img1_features.norm(p=2, dim=-1, keepdim=True)

        img2_features = model.get_image_features(img2_processed_input["pixel_values"].to(device))
        img2_features = img2_features / img2_features.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity between feature vectors
        score = 100 * (img1_features * img2_features).sum(axis=-1)
        return score, len(images1)

    def update(self, images1: Union[Image.Image, List[Image.Image]],
               images2: Union[Image.Image, List[Image.Image]]) -> None:
        """Update CLIP score on a batch of images and text.

        Args:
            images1: Either a single [N, C, H, W] tensor or a list of [C, H, W] tensors
            images2: Either a single [N, C, H, W] tensor or a list of [C, H, W] tensors

        Raises:
            ValueError:
                If not all images have format [C, H, W]
            ValueError:
                If the number of images do not match
        """
        score, n_samples = self._clip_score_update(images1, images2, self.model, self.processor)
        self.score += score.sum(0)
        self.n_samples += n_samples

    def compute(self) -> Tensor:
        """Compute accumulated clip score."""
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

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.multimodal import CLIPScore
            >>> metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
            >>> metric.update(torch.randint(255, (3, 224, 224)), "a photo of a cat")
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.multimodal import CLIPScore
            >>> metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.randint(255, (3, 224, 224)), "a photo of a cat"))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)


class CLIPTScore(Metric):
    r"""Calculates `CLIP Score`_ which is a text-to-image similarity metric.

    CLIP is a reference free metric that can be used to evaluate the correlation between a generated caption for an
    image and the actual content of the image. It has been found to be highly correlated with human judgement. The
    metric is defined as:

    .. math::
        \text{CLIPScore(I, C)} = max(100 * cos(E_I, E_C), 0)

    which corresponds to the cosine similarity between visual CLIP embedding :math:`E_i` for an image :math:`i` and
    textual CLIP embedding :math:`E_C` for an caption :math:`C`. The score is bound between 0 and 100 and the closer
    to 100 the better.

    .. note:: Metric is not scriptable

    Args:
        model_name_or_path: string indicating the version of the CLIP model to use. Available models are:

            - `"openai/clip-vit-base-patch16"`
            - `"openai/clip-vit-base-patch32"`
            - `"openai/clip-vit-large-patch14-336"`
            - `"openai/clip-vit-large-patch14"`

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If transformers package is not installed or version is lower than 4.10.0

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> from torchmetrics.multimodal import CLIPScore
        >>> metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
        >>> score = metric(torch.randint(255, (3, 224, 224)), "a photo of a cat")
        >>> print(score.detach())
        tensor(24.7691)
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
                "openai/clip-vit-base-patch16",
                "openai/clip-vit-base-patch32",
                "openai/clip-vit-large-patch14-336",
                "openai/clip-vit-large-patch14",
            ] = _DEFAULT_MODEL,  # type: ignore[assignment]
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model, self.processor = _get_model_and_processor(model_name_or_path)
        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    @staticmethod
    def _clip_score_update(
            images: Union[Image.Image, List[Image.Image]],
            text: Union[str, List[str]],
            model: _CLIPModel,
            processor: _CLIPProcessor,
    ) -> Tuple[Tensor, int]:
        if len(text) != len(images):
            raise ValueError(
                f"Expected the number of images and text examples to be the same but got {len(images)} and {len(text)}"
            )
        device = next(model.parameters()).device
        processed_input = processor(text=text, images=images, return_tensors="pt", padding=True)

        img_features = model.get_image_features(processed_input["pixel_values"].to(device))
        img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

        txt_features = model.get_text_features(
            processed_input["input_ids"].to(device), processed_input["attention_mask"].to(device)
        )
        txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity between feature vectors
        score = 100 * (img_features * txt_features).sum(axis=-1)
        return score, len(text)

    def update(self, images: Union[Image.Image, List[Image.Image]], text: Union[str, List[str]]) -> None:
        """Update CLIP score on a batch of images and text.

        Args:
            images: Either a single [N, C, H, W] tensor or a list of [C, H, W] tensors
            text: Either a single caption or a list of captions

        Raises:
            ValueError:
                If not all images have format [C, H, W]
            ValueError:
                If the number of images and captions do not match
        """
        score, n_samples = self._clip_score_update(images, text, self.model, self.processor)
        self.score += score.sum(0)
        self.n_samples += n_samples

    def compute(self) -> Tensor:
        """Compute accumulated clip score."""
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

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.multimodal import CLIPScore
            >>> metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
            >>> metric.update(torch.randint(255, (3, 224, 224)), "a photo of a cat")
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.multimodal import CLIPScore
            >>> metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.randint(255, (3, 224, 224)), "a photo of a cat"))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)
