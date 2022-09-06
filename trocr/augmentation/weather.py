import cv2
import numpy as np
import math
from PIL import Image, ImageOps, ImageDraw
from skimage import color
from scipy import interpolate
from pkg_resources import resource_filename
from io import BytesIO
from .ops import plasma_fractal, clipped_zoom, MotionImage

'''
    PIL resize (W,H)
'''
class Fog:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        c = [(1.5, 2), (2., 2), (2.5, 1.7)]
        if mag<0 or mag>=len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = np.array(img) / 255.
        max_val = img.max()
        fog = c[0] * plasma_fractal(wibbledecay=c[1])[:H, :W][..., np.newaxis]
        #x += c[0] * plasma_fractal(wibbledecay=c[1])[:224, :224][..., np.newaxis]
        #return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255
        if isgray:
            fog = np.squeeze(fog)
        else:
            fog = np.repeat(fog, 3, axis=2)

        # print('img', img.shape)
        # print('fog', fog.shape)
        # print(H, W)
        # exit(0)
        fog = cv2.resize(fog, dsize=(H, W), interpolation=cv2.INTER_CUBIC)
        img += fog
        img = np.clip(img * max_val / (max_val + c[0]), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


class Frost:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7)]
        if mag<0 or mag>=len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]

        filename = [resource_filename(__name__, 'frost/frost1.png'),
                    resource_filename(__name__, 'frost/frost2.png'),
                    resource_filename(__name__, 'frost/frost3.png'),
                    resource_filename(__name__, 'frost/frost4.jpg'),
                    resource_filename(__name__, 'frost/frost5.jpg'),
                    resource_filename(__name__, 'frost/frost6.jpg')
                    ]
        index = np.random.randint(0, len(filename))
        filename = filename[index]
        frost = cv2.imread(filename)
        frost = cv2.resize(frost, dsize=(H, W), interpolation=cv2.INTER_CUBIC)
        #randomly crop and convert to rgb
        # x_start, y_start = np.random.randint(0, frost.shape[0] - H), np.random.randint(0, frost.shape[1] - W)
        x_start = 0
        y_start = 0
        frost = frost[x_start:x_start + H, y_start:y_start + W][..., [2, 1, 0]]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = np.array(img)
        
        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)

        img = img * c[0]
        frost = frost * c[1]
        img = np.clip(c[0] * img + c[1] * frost, 0, 255)
        img = Image.fromarray(img.astype(np.uint8))
        if isgray:
            img = ImageOps.grayscale(img)

        return img

class Snow:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
             (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
             (0.55, 0.3, 4, 0.9, 12, 8, 0.7)]
        if mag<0 or mag>=len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = np.array(img, dtype=np.float32) / 255.
        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)

        snow_layer = np.random.normal(size=img.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

        #snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
        snow_layer[snow_layer < c[3]] = 0

        snow_layer = Image.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
        output = BytesIO()
        snow_layer.save(output, format='PNG')
        snow_layer = MotionImage(blob=output.getvalue())

        snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

        snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                                  cv2.IMREAD_UNCHANGED) / 255.

        #snow_layer = cv2.cvtColor(snow_layer, cv2.COLOR_BGR2RGB)

        snow_layer = snow_layer[..., np.newaxis]

        img = c[6] * img
        gray_img = (1 - c[6]) * np.maximum(img, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(H, W, 1) * 1.5 + 0.5)
        img += gray_img
        img = np.clip(img + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255
        img = Image.fromarray(img.astype(np.uint8))
        if isgray:
            img = ImageOps.grayscale(img)

        return img

class Rain:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        img = img.copy()
        W, H = img.size
        n_channels = len(img.getbands())
        isgray = n_channels == 1
        line_width = np.random.randint(1, 2)

        c =[50, 70, 90]
        if mag<0 or mag>=len(c):
            index = 0
        else:
            index = mag
        c = c[index]

        n_rains = np.random.randint(c, c+20)
        slant = np.random.randint(-60, 60)
        fillcolor = 200 if isgray else (200,200,200)

        draw = ImageDraw.Draw(img)
        for i in range(1, n_rains):
            length = np.random.randint(5, 10)
            x1 = np.random.randint(0, W-length)
            y1 = np.random.randint(0, H-length)
            x2 = x1 + length*math.sin(slant*math.pi/180.)
            y2 = y1 + length*math.cos(slant*math.pi/180.)
            x2 = int(x2)
            y2 = int(y2)
            draw.line([(x1,y1), (x2,y2)], width=line_width, fill=fillcolor)

        return img

class Shadow:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        #img = img.copy()
        W, H = img.size
        n_channels = len(img.getbands())
        isgray = n_channels == 1

        c =[64, 96, 128]
        if mag<0 or mag>=len(c):
            index = 0
        else:
            index = mag
        c = c[index]

        img = img.convert('RGBA')
        overlay = Image.new('RGBA', img.size, (255,255,255,0))
        draw = ImageDraw.Draw(overlay) 
        transparency = np.random.randint(c, c+32)
        x1 = np.random.randint(0, W//2)
        y1 = 0

        x2 = np.random.randint(W//2, W)
        y2 = 0

        x3 = np.random.randint(W//2, W)
        y3 = H - 1

        x4 = np.random.randint(0, W//2)
        y4 = H - 1

        draw.polygon([(x1,y1), (x2,y2), (x3,y3), (x4,y4)], fill=(0,0,0,transparency))

        img = Image.alpha_composite(img, overlay)
        img = img.convert("RGB")
        if isgray:
            img = ImageOps.grayscale(img)

        return img
