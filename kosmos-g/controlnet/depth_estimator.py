import PIL.Image
import numpy as np
from controlnet_aux.util import HWC3
from transformers import pipeline

from controlnet.cv_utils import resize_image


class DepthEstimator:
    def __init__(self):
        self.model = pipeline('depth-estimation')

    def __call__(self, image: np.ndarray, **kwargs) -> PIL.Image.Image:
        detect_resolution = kwargs.pop('detect_resolution', 512)
        image_resolution = kwargs.pop('image_resolution', 512)
        image = np.array(image)
        image = HWC3(image)
        image = resize_image(image, resolution=detect_resolution)
        image = PIL.Image.fromarray(image)
        image = self.model(image)
        image = image['depth']
        image = np.array(image)
        image = HWC3(image)
        image = resize_image(image, resolution=image_resolution)
        return PIL.Image.fromarray(image)
