from PIL import Image
import cv2
import base64
import io
import numpy as np


def convert(data):
    if isinstance(data, dict):
        ndata = {}
        for key, value in data.items():
            nkey = key.decode()
            if nkey == 'img':
                img = Image.open(io.BytesIO(value))
                img = img.convert('RGB')
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                nvalue = img
            else:
                nvalue = convert(value)
            ndata[nkey] = nvalue
        return ndata
    elif isinstance(data, list):
        return [convert(item) for item in data]
    elif isinstance(data, bytes):
        return data.decode()
    else:
        return data


def to_np(x):
    return x.cpu().data.numpy()
