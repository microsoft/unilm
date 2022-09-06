
import cv2
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from wand.image import Image as WandImage
from scipy.ndimage import zoom as scizoom
from skimage.filters import gaussian
from wand.api import library as wandlibrary
from io import BytesIO

#from skimage import color
from .ops import MotionImage, clipped_zoom, disk, plasma_fractal
'''
    PIL resize (W,H)
'''
class GaussianBlur:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        #kernel = [(31,31)] prev 1 level only
        kernel = (31, 31)
        sigmas = [.5, 1, 2]
        if mag<0 or mag>=len(kernel):
            index = np.random.randint(0, len(sigmas))
        else:
            index = mag

        sigma = sigmas[index]
        return transforms.GaussianBlur(kernel_size=kernel, sigma=sigma)(img)


class DefocusBlur:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        n_channels = len(img.getbands())
        isgray = n_channels == 1
        #c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)]
        c = [(2, 0.1), (3, 0.1), (4, 0.1)] #, (6, 0.5)] #prev 2 levels only
        if mag<0 or mag>=len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]

        img = np.array(img) / 255.
        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)
            n_channels = 3
        kernel = disk(radius=c[0], alias_blur=c[1])

        channels = []
        for d in range(n_channels):
            channels.append(cv2.filter2D(img[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3
        
        #if isgray:
        #    img = img[:,:,0]
        #    img = np.squeeze(img)

        img = np.clip(channels, 0, 1) * 255
        img = Image.fromarray(img.astype(np.uint8))
        if isgray:
            img = ImageOps.grayscale(img)

        return img


class MotionBlur:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        n_channels = len(img.getbands())
        isgray = n_channels == 1
        #c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)]
        c = [(10, 3), (12, 4), (14, 5)]
        if mag<0 or mag>=len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]

        output = BytesIO()
        img.save(output, format='PNG')
        img = MotionImage(blob=output.getvalue())

        img.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))
        img = cv2.imdecode(np.fromstring(img.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img.astype(np.uint8))

        if isgray:
            img = ImageOps.grayscale(img)

        return img

class GlassBlur:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        #c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
        c = [(0.7, 1, 2), (0.75, 1, 2), (0.8, 1, 2)] #, (1, 2, 3)] #prev 2 levels only
        if mag<0 or mag>=len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag

        c = c[index]

        img = np.uint8(gaussian(np.array(img) / 255., sigma=c[0], multichannel=True) * 255)

        # locally shuffle pixels
        for i in range(c[2]):
            for h in range(H - c[1], c[1], -1):
                for w in range(W - c[1], c[1], -1):
                    dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                    h_prime, w_prime = h + dy, w + dx
                    # swap
                    img[h, w], img[h_prime, w_prime] = img[h_prime, w_prime], img[h, w]

        img = np.clip(gaussian(img / 255., sigma=c[0], multichannel=True), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


class ZoomBlur:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        c = [np.arange(1, 1.11, .01),
             np.arange(1, 1.16, .01),
             np.arange(1, 1.21, .02)]
        if mag<0 or mag>=len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag

        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        uint8_img = img
        img = (np.array(img) / 255.).astype(np.float32)

        out = np.zeros_like(img)
        for zoom_factor in c:
            ZW = int(W*zoom_factor)
            ZH = int(H*zoom_factor)
            zoom_img = uint8_img.resize((ZW, ZH), Image.BICUBIC)
            x1 = (ZW - W) // 2
            y1 = (ZH - H) // 2
            x2 = x1 + W
            y2 = y1 + H
            zoom_img = zoom_img.crop((x1,y1,x2,y2))
            out += (np.array(zoom_img) / 255.).astype(np.float32)

        img = (img + out) / (len(c) + 1)

        img = np.clip(img, 0, 1) * 255
        img = Image.fromarray(img.astype(np.uint8))

        return img

