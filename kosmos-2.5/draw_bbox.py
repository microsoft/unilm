import os
import json
import argparse
from PIL import Image, ImageDraw

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Input image")
    parser.add_argument("--ocr_json_path", type=str, required=True, help="OCR Json Path")
    parser.add_argument("--out", type=str, default="", help="Output path")
    parser.add_argument("--line_width", type=int, default=1, help="line width")
    args = parser.parse_args()

    assert os.path.exists(args.image), "Image does not exist."
    assert os.path.exists(args.ocr_json_path), "OCR json does not exist."

    if args.out == "":
        args.out = os.path.join("./", f"res_{os.path.basename(args.image)}")

    return args

def draw_bbox(img_path, ocr_res, save_path, line_width):
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for result in ocr_res['results']:
        x0, y0, x1, y1 = result['bounding box']['x0'], result['bounding box']['y0'], result['bounding box']['x1'], \
                         result['bounding box']['y1']

        draw.line([(x0, y0), (x0, y1)], fill='red', width=line_width)
        draw.line([(x0, y1), (x1, y1)], fill='red', width=line_width)
        draw.line([(x1, y1), (x1, y0)], fill='red', width=line_width)
        draw.line([(x1, y0), (x0, y0)], fill='red', width=line_width)
    image.save(save_path)


def load_ocr_result(ocr_json_path):
    with open(ocr_json_path, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    return data


if __name__ == '__main__':
    args = get_args()

    ocr = load_ocr_result(args.ocr_json_path)
    draw_bbox(args.image, ocr, args.out, args.line_width)
    print('done')






