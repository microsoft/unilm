
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

'''
    PIL resize (W,H)
    Torch resize is (H,W)
'''
class VGrid:
    def __init__(self):
        pass

    def __call__(self, img, copy=True, max_width=4, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        if copy:
            img = img.copy()
        W, H = img.size

        if mag<0 or mag>max_width:
            line_width = np.random.randint(1, max_width)
            image_stripe = np.random.randint(1, max_width)
        else:
            line_width = 1
            image_stripe = 3 - mag

        n_lines = W // (line_width + image_stripe) + 1
        draw = ImageDraw.Draw(img)
        for i in range(1, n_lines):
            x = image_stripe*i + line_width*(i-1)
            draw.line([(x,0), (x,H)], width=line_width, fill='black')

        return img

class HGrid:
    def __init__(self):
        pass

    def __call__(self, img, copy=True, max_width=4, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        if copy:
            img = img.copy()
        W, H = img.size
        if mag<0 or mag>max_width:
            line_width = np.random.randint(1, max_width)
            image_stripe = np.random.randint(1, max_width)
        else:
            line_width = 1
            image_stripe = 3 - mag

        n_lines = H // (line_width + image_stripe) + 1
        draw = ImageDraw.Draw(img)
        for i in range(1, n_lines):
            y = image_stripe*i + line_width*(i-1)
            draw.line([(0,y), (W, y)], width=line_width, fill='black')

        return img

class Grid:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        img = VGrid()(img, copy=True, mag=mag)
        img = HGrid()(img, copy=False, mag=mag)
        return img

class RectGrid:
    def __init__(self):
        pass

    def __call__(self, img, isellipse=False, mag=-1,  prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        img = img.copy()
        W, H = img.size
        line_width = 1 
        image_stripe = 3 - mag #np.random.randint(2, 6)
        offset = 4 if isellipse else 1
        n_lines = ((H//2) // (line_width + image_stripe)) + offset
        draw = ImageDraw.Draw(img)
        x_center = W // 2
        y_center = H // 2
        for i in range(1, n_lines):
            dx = image_stripe*i + line_width*(i-1)
            dy = image_stripe*i + line_width*(i-1)
            x1 = x_center - (dx * W//H)
            y1 = y_center - dy
            x2 = x_center + (dx * W/H) 
            y2 = y_center + dy
            if isellipse:
                draw.ellipse([(x1,y1), (x2, y2)], width=line_width, outline='black')
            else:
                draw.rectangle([(x1,y1), (x2, y2)], width=line_width, outline='black')

        return img

class EllipseGrid:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        img = RectGrid()(img, isellipse=True, mag=mag, prob=prob)
        return img
