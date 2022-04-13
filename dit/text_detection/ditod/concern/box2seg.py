import cv2
import numpy as np
from scipy import interpolate

def intersection(x, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x2 == x1:
        return 0
    k = (x - x1) / (x2 - x1)
    return k * (y2 - y1) + y1


def midpoint(p1, p2, typed=float):
    return [typed((p1[0] + p2[0]) / 2), typed((p1[1] + p2[1]) / 2)]


def resize_with_coordinates(image, width, height, coordinates):
    original_height, original_width = image.shape[:2]
    resized_image = cv2.resize(image, (width, height))
    if coordinates is not None:
        assert coordinates.ndim == 2
        assert coordinates.shape[-1] == 2

        rate_x = width / original_width
        rate_y = height / original_height

        coordinates = coordinates * (rate_x, rate_y)
    return resized_image, coordinates


def box2seg(image, boxes, label):
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    seg = np.zeros((height, width), dtype=np.float32)
    points = []
    for box_index in range(boxes.shape[0]):
        box = boxes[box_index, :, :] # 4x2
        left_top = box[0]
        right_top = box[1]
        right_bottom = box[2]
        left_bottom = box[3]

        left = [(left_top[0] + left_bottom[0]) / 2, (left_top[1] + left_bottom[1]) / 2]
        right = [(right_top[0] + right_bottom[0]) / 2, (right_top[1] + right_bottom[1]) / 2]

        center = midpoint(left, right)
        points.append(midpoint(left, center))
        points.append(midpoint(right, center))

        poly = np.array([midpoint(left_top, center),
            midpoint(right_top, center),
            midpoint(right_bottom, center),
            midpoint(left_bottom, center)
            ])
        seg = cv2.fillPoly(seg, [poly.reshape(4, 1, 2).astype(np.int32)], int(label[box_index]))

    left_y = intersection(0, points[0], points[1])
    right_y = intersection(width, points[-1], points[-2])
    points.insert(0, [0, left_y])
    points.append([width, right_y])
    points = np.array(points)

    f = interpolate.interp1d(points[:, 0], points[:, 1], fill_value='extrapolate')
    xnew = np.arange(0, width, 1)
    ynew = f(xnew).clip(0, height-1)
    for x in range(width - 1):
        mask[int(ynew[x]), x] = 1
    return ynew.reshape(1, -1).round(), seg
