#!/usr/bin/env mdl
import socket
import base64
import cv2
import numpy as np
from collections import OrderedDict

from .server import get_server


def jpeg_encode(img):
    return cv2.imencode('.png', img)[1]


def get_free_port(rng, low=2000, high=10000):
    in_use = True
    while in_use:
        port = rng.randint(high - low) + low
        in_use = False
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("0.0.0.0", port))
        except socket.error as e:
            if e.errno == 98:  # port already in use
                in_use = True
        s.close()
    return port


class Manager:
    def __init__(self, img_encode_method=jpeg_encode, rng=None):
        self._queue = OrderedDict()
        self._server = None
        self.img_encode_method = img_encode_method
        if rng is None:
            rng = np.random.RandomState(self.get_default_seed())
        self.rng = rng

    def get_default_seed(self):
        return 0

    def imshow(self, title, img):
        data = self.img_encode_method(img)
        data = base64.b64encode(data)
        data = data.decode('utf8')
        self._queue[title] = data

    def waitKey(self, delay=0):
        if self._server is None:
            self.port = get_free_port(self.rng)
            self._server, self._conn = get_server(port=self.port)
        self._conn.send([delay, list(self._queue.items())])
        # self._queue = OrderedDict()
        return self._conn.recv()

global_manager = Manager()

