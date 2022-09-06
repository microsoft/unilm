
import os
import cv2
from warp import Curve, Distort, Stretch
from geometry import Rotate, Perspective, Shrink, TranslateX, TranslateY
from pattern import VGrid, HGrid, Grid, RectGrid, EllipseGrid
from noise import GaussianNoise, ShotNoise, ImpulseNoise, SpeckleNoise
from blur import GaussianBlur, DefocusBlur, MotionBlur, GlassBlur, ZoomBlur
from camera import Contrast, Brightness, JpegCompression, Pixelate
from weather import Fog, Snow, Frost, Rain, Shadow
from process import Posterize, Solarize, Invert, Equalize, AutoContrast, Sharpness, Color

from PIL import Image
import PIL.ImageOps
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default="images/delivery.png", help='Load image file')
    parser.add_argument('--results', default="results", help='Load image file')
    parser.add_argument('--gray', action='store_true', help='Convert to grayscale 1st')
    opt = parser.parse_args()
    os.makedirs(opt.results, exist_ok=True)

    img = Image.open(opt.image)
    img = img.resize( (100,32) )
    ops = [Curve(), Rotate(), Perspective(), Distort(), Stretch(), Shrink(), TranslateX(), TranslateY(), VGrid(), HGrid(), Grid(), RectGrid(), EllipseGrid()]
    ops.extend([GaussianNoise(), ShotNoise(), ImpulseNoise(), SpeckleNoise()])
    ops.extend([GaussianBlur(), DefocusBlur(), MotionBlur(), GlassBlur(), ZoomBlur()])
    ops.extend([Contrast(), Brightness(), JpegCompression(), Pixelate()])
    ops.extend([Fog(), Snow(), Frost(), Rain(), Shadow()])
    ops.extend([Posterize(), Solarize(), Invert(), Equalize(), AutoContrast(), Sharpness(), Color()])
    for op in ops:
        for mag in range(3):
            filename = type(op).__name__ + "-" + str(mag) + ".png"
            out_img = op(img, mag=mag)
            if opt.gray:
                out_img = PIL.ImageOps.grayscale(out_img)
            out_img.save(os.path.join(opt.results, filename))


