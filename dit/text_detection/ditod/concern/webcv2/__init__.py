#!/usr/bin/env mdl
class WebCV2:
    def __init__(self):
        import cv2
        self._cv2 = cv2
        from .manager import global_manager as gm
        self._gm = gm

    def __getattr__(self, name):
        if hasattr(self._gm, name):
            return getattr(self._gm, name)
        elif hasattr(self._cv2, name):
            return getattr(self._cv2, name)
        else:
            raise AttributeError

import sys
sys.modules[__name__] = WebCV2()

