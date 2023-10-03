import gc

import PIL.Image
import numpy as np
import torch
from controlnet_aux import (CannyDetector, ContentShuffleDetector, HEDdetector, LineartAnimeDetector, LineartDetector,
                            MidasDetector, MLSDdetector, NormalBaeDetector, OpenposeDetector, PidiNetDetector)
from controlnet_aux.util import HWC3

from controlnet.cv_utils import resize_image
from controlnet.depth_estimator import DepthEstimator
from controlnet.image_segmentor import ImageSegmentor


class ControlNet_Preprocessor:
    MODEL_ID = 'lllyasviel/Annotators'

    def __init__(self):
        self.model = None
        self.name = ''

    def load(self, name: str) -> None:
        if name == self.name:
            return
        if name == 'HED':
            self.model = HEDdetector.from_pretrained(self.MODEL_ID)
        elif name == 'Midas':
            self.model = MidasDetector.from_pretrained(self.MODEL_ID)
        elif name == 'MLSD':
            self.model = MLSDdetector.from_pretrained(self.MODEL_ID)
        elif name == 'Openpose':
            self.model = OpenposeDetector.from_pretrained(self.MODEL_ID)
        elif name == 'PidiNet':
            self.model = PidiNetDetector.from_pretrained(self.MODEL_ID)
        elif name == 'NormalBae':
            self.model = NormalBaeDetector.from_pretrained(self.MODEL_ID)
        elif name == 'Lineart':
            self.model = LineartDetector.from_pretrained(self.MODEL_ID)
        elif name == 'LineartAnime':
            self.model = LineartAnimeDetector.from_pretrained(self.MODEL_ID)
        elif name == 'Canny':
            self.model = CannyDetector()
        elif name == 'ContentShuffle':
            self.model = ContentShuffleDetector()
        elif name == 'DPT':
            self.model = DepthEstimator()
        elif name == 'UPerNet':
            self.model = ImageSegmentor()
        else:
            raise ValueError
        torch.cuda.empty_cache()
        gc.collect()
        self.name = name

    def __call__(self, image: PIL.Image.Image, **kwargs) -> PIL.Image.Image:
        if self.name == 'Canny':
            if 'detect_resolution' in kwargs:
                detect_resolution = kwargs.pop('detect_resolution')
                image = np.array(image)
                image = HWC3(image)
                image = resize_image(image, resolution=detect_resolution)
            image = self.model(image, **kwargs)
            return PIL.Image.fromarray(image)
        elif self.name == 'Midas':
            detect_resolution = kwargs.pop('detect_resolution', 512)
            image_resolution = kwargs.pop('image_resolution', 512)
            image = np.array(image)
            image = HWC3(image)
            image = resize_image(image, resolution=detect_resolution)
            image = self.model(image, **kwargs)
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            return PIL.Image.fromarray(image)
        else:
            image = np.array(image)
            return self.model(image, **kwargs)

    @torch.inference_mode()
    def preprocess_canny(self, image, image_resolution, low_threshold, high_threshold):
        self.load('Canny')
        control_image = self(
            image=image,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            detect_resolution=image_resolution
        )
        return control_image

    @torch.inference_mode()
    def preprocess_mlsd(self, image, image_resolution, preprocess_resolution, value_threshold, distance_threshold):
        self.load('MLSD')
        control_image = self(
            image=image,
            image_resolution=image_resolution,
            detect_resolution=preprocess_resolution,
            thr_v=value_threshold,
            thr_d=distance_threshold,
        )
        return control_image

    @torch.inference_mode()
    def preprocess_scribble(self, image, image_resolution, preprocess_resolution, preprocessor_name):
        if preprocessor_name == 'None':
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        elif preprocessor_name == 'HED':
            self.load(preprocessor_name)
            control_image = self(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                scribble=False,
            )
        elif preprocessor_name == 'PidiNet':
            self.load(preprocessor_name)
            control_image = self(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                safe=False,
            )
        else:
            raise ValueError
        return control_image

    @torch.inference_mode()
    def preprocess_scribble_interactive(self, image_and_mask, image_resolution):
        image = image_and_mask['mask']
        image = HWC3(image)
        image = resize_image(image, resolution=image_resolution)
        control_image = PIL.Image.fromarray(image)
        return control_image

    @torch.inference_mode()
    def preprocess_softedge(self, image, image_resolution, preprocess_resolution, preprocessor_name):
        if preprocessor_name == 'None':
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        elif preprocessor_name in ['HED', 'HED safe']:
            safe = 'safe' in preprocessor_name
            self.load('HED')
            control_image = self(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                scribble=safe,
            )
        elif preprocessor_name in ['PidiNet', 'PidiNet safe']:
            safe = 'safe' in preprocessor_name
            self.load('PidiNet')
            control_image = self(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                safe=safe,
            )
        else:
            raise ValueError
        return control_image

    @torch.inference_mode()
    def preprocess_openpose(self, image, image_resolution, preprocess_resolution, preprocessor_name):
        if preprocessor_name == 'None':
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        else:
            self.load('Openpose')
            control_image = self(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                hand_and_face=True,
            )
        return control_image

    @torch.inference_mode()
    def preprocess_segmentation(self, image, image_resolution, preprocess_resolution, preprocessor_name):
        if preprocessor_name == 'None':
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        else:
            self.load(preprocessor_name)
            control_image = self(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
            )
        return control_image

    @torch.inference_mode()
    def preprocess_depth(self, image, image_resolution, preprocess_resolution, preprocessor_name):
        if preprocessor_name == 'None':
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        else:
            self.load(preprocessor_name)
            control_image = self(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
            )
        return control_image

    @torch.inference_mode()
    def preprocess_normal(self, image, image_resolution, preprocess_resolution, preprocessor_name):
        if preprocessor_name == 'None':
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        else:
            self.load('NormalBae')
            control_image = self(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
            )
        return control_image

    @torch.inference_mode()
    def preprocess_lineart(self, image, image_resolution, preprocess_resolution, preprocessor_name):
        if preprocessor_name in ['None', 'None (anime)']:
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        elif preprocessor_name in ['Lineart', 'Lineart coarse']:
            coarse = 'coarse' in preprocessor_name
            self.load('Lineart')
            control_image = self(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                coarse=coarse,
            )
        elif preprocessor_name == 'Lineart (anime)':
            self.load('LineartAnime')
            control_image = self(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
            )
        else:
            raise ValueError
        return control_image

    @torch.inference_mode()
    def preprocess_shuffle(self, image, image_resolution, preprocessor_name):
        if preprocessor_name == 'None':
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        else:
            self.load(preprocessor_name)
            control_image = self(
                image=image,
                image_resolution=image_resolution,
            )
        return control_image

    @torch.inference_mode()
    def preprocess_ip2p(self, image, image_resolution):
        image = HWC3(image)
        image = resize_image(image, resolution=image_resolution)
        control_image = PIL.Image.fromarray(image)
        return control_image
