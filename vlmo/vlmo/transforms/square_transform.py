# code in this file is adpated from the ALBEF repo (https://github.com/salesforce/ALBEF)

from .utils import (
    inception_normalize,
)
from torchvision import transforms
from .randaugment import RandomAugment
from PIL import Image


def square_transform(size=224):
    return transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )


def square_transform_randaug(size=224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            inception_normalize,
        ]
    )