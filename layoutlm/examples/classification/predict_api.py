import os, uuid, pathlib, base64
import sys
ROOT_DIR = os.path.abspath("../../../")
sys.path.append(ROOT_DIR)
from layoutlm.data.convert import convert_img_to_xml
from examples.classification.predict import make_prediction

# assumes model exists and is in same directory as this file
MODEL_DIR = 'aetna-trained-model'
OUTPUT_DIR = 'output'


def predict(base64_img):
    try:
        os.mkdir(OUTPUT_DIR)
    except:
        pass
    filename = uuid.uuid4().hex
    # assumes that base64_img encodes a .tiff file
    img = os.path.join(OUTPUT_DIR, filename + '.tiff')
    with open(img, 'wb') as file_to_save:
        decoded_image_data = base64.b64decode(base64_img, '-_')
        file_to_save.write(decoded_image_data)
    convert_img_to_xml(img, OUTPUT_DIR)
    hocr = os.path.join(OUTPUT_DIR, filename + '.xml')
    label, confidence = make_prediction(MODEL_DIR, hocr)
    response = {
        'label': label,
        'confidence': confidence
    }
    return response

if __name__ == "__main__":
    # cob-1-1.png
    predict('')