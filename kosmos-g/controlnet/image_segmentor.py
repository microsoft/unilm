import PIL.Image
import cv2
import numpy as np
import torch
from controlnet_aux.util import HWC3, ade_palette
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

from controlnet.cv_utils import resize_image


class ImageSegmentor:
    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained(
            'openmmlab/upernet-convnext-small')
        self.image_segmentor = UperNetForSemanticSegmentation.from_pretrained(
            'openmmlab/upernet-convnext-small')

    @torch.inference_mode()
    def __call__(self, image: np.ndarray, **kwargs) -> PIL.Image.Image:
        detect_resolution = kwargs.pop('detect_resolution', 512)
        image_resolution = kwargs.pop('image_resolution', 512)
        image = HWC3(image)
        image = resize_image(image, resolution=detect_resolution)
        image = PIL.Image.fromarray(image)

        pixel_values = self.image_processor(image,
                                            return_tensors='pt').pixel_values
        outputs = self.image_segmentor(pixel_values)
        seg = self.image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]])[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(ade_palette()):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)

        color_seg = resize_image(color_seg,
                                 resolution=image_resolution,
                                 interpolation=cv2.INTER_NEAREST)
        return PIL.Image.fromarray(color_seg)
