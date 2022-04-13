#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : visualizer.py
# Author            : Zhaoyi Wan <wanzhaoyi@megvii.com>
# Date              : 08.01.2019
# Last Modified Date: 02.12.2019
# Last Modified By  : Minghui Liao 
import torch
import numpy as np
import cv2

class Visualize:
    @classmethod
    def visualize(cls, x):
        dimension = len(x.shape)
        if dimension == 2:
            pass
        elif dimension == 3:
            pass

    @classmethod
    def to_np(cls, x):
        return x.cpu().data.numpy()

    @classmethod
    def visualize_weights(cls, tensor, format='HW', normalize=True):
        if isinstance(tensor, torch.Tensor):
            x = cls.to_np(tensor.permute(format.index('H'), format.index('W')))
        else:
            x = tensor.transpose(format.index('H'), format.index('W'))
        if normalize:
            x = (x - x.min()) / (x.max() - x.min())
        # return np.tile(x * 255., (3, 1, 1)).swapaxes(0, 2).swapaxes(1, 0).astype(np.uint8)
        return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_JET)

    @classmethod
    def visualize_points(cls, image, tensor, radius=5, normalized=True):
        if isinstance(tensor, torch.Tensor):
            points = cls.to_np(tensor)
        else:
            points = tensor
        if normalized:
            points = points * image.shape[:2][::-1]
        for i in range(points.shape[0]):
            color = np.random.randint(
                0, 255, (3, ), dtype=np.uint8).astype(np.float)
            image = cv2.circle(image,
                               tuple(points[i].astype(np.int32).tolist()),
                               radius, color, thickness=radius//2)
        return image

    @classmethod
    def visualize_heatmap(cls, tensor, format='CHW'):
        if isinstance(tensor, torch.Tensor):
            x = cls.to_np(tensor.permute(format.index('H'),
                                         format.index('W'), format.index('C')))
        else:
            x = tensor.transpose(
                format.index('H'), format.index('W'), format.index('C'))
        canvas = np.zeros((x.shape[0], x.shape[1], 3), dtype=np.float)

        for c in range(0, x.shape[-1]):
            color = np.random.randint(
                0, 255, (3, ), dtype=np.uint8).astype(np.float)
            canvas += np.tile(x[:, :, c], (3, 1, 1)
                              ).swapaxes(0, 2).swapaxes(1, 0) * color

        canvas = canvas.astype(np.uint8)
        return canvas

    @classmethod
    def visualize_classes(cls, x):
        canvas = np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)
        for c in range(int(x.max())):
            color = np.random.randint(
                0, 255, (3, ), dtype=np.uint8).astype(np.float)
            canvas[np.where(x == c)] = color
        return canvas

    @classmethod
    def visualize_grid(cls, x, y, stride=16, color=(0, 0, 255), canvas=None):
        h, w = x.shape
        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
        # canvas = np.concatenate([canvas, canvas], axis=1)
        i, j = 0, 0
        while i < w:
            j = 0
            while j < h:
                canvas = cv2.circle(canvas, (int(x[i, j] * w + 0.5), int(y[i, j] * h + 0.5)), radius=max(stride//4, 1), color=color, thickness=stride//8)
                j += stride
            i += stride
        return canvas

    @classmethod
    def visualize_rect(cls, canvas, _rect, color=(0, 0, 255)):
        rect = (_rect + 0.5).astype(np.int32)
        return cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), color)
