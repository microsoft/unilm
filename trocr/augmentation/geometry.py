
import cv2
import numpy as np
from PIL import Image, ImageOps

'''
    PIL resize (W,H)
    Torch resize is (H,W)
'''
class Shrink:
    def __init__(self):
        self.tps = cv2.createThinPlateSplineShapeTransformer()
        self.translateXAbs = TranslateXAbs()
        self.translateYAbs = TranslateYAbs()

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size
        img = np.array(img)
        srcpt = list()
        dstpt = list()

        W_33 = 0.33 * W
        W_50 = 0.50 * W
        W_66 = 0.66 * W

        H_50 = 0.50 * H

        P = 0

        #frac = 0.4

        b = [.2, .3, .4]
        if mag<0 or mag>=len(b):
            index = 0
        else:
            index = mag
        frac = b[index]

        # left-most
        srcpt.append([P, P])
        srcpt.append([P, H-P])
        x = np.random.uniform(frac-.1, frac)*W_33 
        y = np.random.uniform(frac-.1, frac)*H_50
        dstpt.append([P+x, P+y])
        dstpt.append([P+x, H-P-y])
        
        # 2nd left-most 
        srcpt.append([P+W_33, P])
        srcpt.append([P+W_33, H-P])
        dstpt.append([P+W_33, P+y])
        dstpt.append([P+W_33, H-P-y])
        
        # 3rd left-most 
        srcpt.append([P+W_66, P])
        srcpt.append([P+W_66, H-P])
        dstpt.append([P+W_66, P+y])
        dstpt.append([P+W_66, H-P-y])
        
        # right-most 
        srcpt.append([W-P, P])
        srcpt.append([W-P, H-P])
        dstpt.append([W-P-x, P+y])
        dstpt.append([W-P-x, H-P-y])

        N = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(N)]
        dst_shape = np.array(dstpt).reshape((-1, N, 2))
        src_shape = np.array(srcpt).reshape((-1, N, 2))
        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)
        img = Image.fromarray(img)

        if np.random.uniform(0, 1) < 0.5:
            img = self.translateXAbs(img, val=x)
        else:
            img = self.translateYAbs(img, val=y)

        return img


class Rotate:
    def __init__(self, square_side=224):
        self.side = square_side

    def __call__(self, img, iscurve=False, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size

        if H!=self.side or W!=self.side:
            img = img.resize((self.side, self.side), Image.BICUBIC)

        b = [20., 40, 60]
        if mag<0 or mag>=len(b):
            index = 1
        else:
            index = mag
        rotate_angle = b[index]

        angle = np.random.uniform(rotate_angle-20, rotate_angle)
        if np.random.uniform(0, 1) < 0.5:
            angle = -angle

        #angle = np.random.normal(loc=0., scale=rotate_angle)
        #angle = min(angle, 2*rotate_angle)
        #angle = max(angle, -2*rotate_angle)

        expand = False if iscurve else True
        img = img.rotate(angle=angle, resample=Image.BICUBIC, expand=expand)
        img = img.resize((W, H), Image.BICUBIC)

        return img

class Perspective:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        W, H = img.size

        # upper-left, upper-right, lower-left, lower-right
        src =  np.float32([[0, 0], [W, 0], [0, H], [W, H]])
        #low = 0.3 

        b = [.1, .2, .3]
        if mag<0 or mag>=len(b):
            index = 2
        else:
            index = mag
        low = b[index]

        high = 1 - low
        if np.random.uniform(0, 1) > 0.5:
            toprightY = np.random.uniform(low, low+.1)*H
            bottomrightY = np.random.uniform(high-.1, high)*H
            dest = np.float32([[0, 0], [W, toprightY], [0, H], [W, bottomrightY]])
        else:
            topleftY = np.random.uniform(low, low+.1)*H
            bottomleftY = np.random.uniform(high-.1, high)*H
            dest = np.float32([[0, topleftY], [W, 0], [0, bottomleftY], [W, H]])
        M = cv2.getPerspectiveTransform(src, dest)
        img = np.array(img)
        img = cv2.warpPerspective(img, M, (W, H) )
        img = Image.fromarray(img)

        return img


class TranslateX:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        b = [.03, .06, .09]
        if mag<0 or mag>=len(b):
            index = 2
        else:
            index = mag
        v = b[index]
        v = np.random.uniform(v-0.03, v)

        v = v * img.size[0]
        if np.random.uniform(0,1) > 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


class TranslateY:
    def __init__(self):
        pass

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        b = [.07, .14, .21]
        if mag<0 or mag>=len(b):
            index = 2
        else:
            index = mag
        v = b[index]
        v = np.random.uniform(v-0.07, v)

        v = v * img.size[1]
        if np.random.uniform(0,1) > 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


class TranslateXAbs:
    def __init__(self):
        pass

    def __call__(self, img, val=0, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        v = np.random.uniform(0, val)

        if np.random.uniform(0,1) > 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


class TranslateYAbs:
    def __init__(self):
        pass

    def __call__(self, img, val=0, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img

        v = np.random.uniform(0, val)

        if np.random.uniform(0,1) > 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))






