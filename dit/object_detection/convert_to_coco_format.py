import os
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import json
from PIL import Image
from shutil import copyfile


def convert(ROOT, TRACK, SPLIT):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "table"}, ],
    }
    DATA_DIR = f"{ROOT}/{TRACK}/{SPLIT}"
    prefix = "cTDaR_t0" if TRACK == "trackA_archival" else "cTDaR_t1"
    print(TRACK, SPLIT, prefix)
    table_count = 0
    for file in sorted(os.listdir(DATA_DIR)):
        if file.startswith(prefix) and file.endswith(".jpg"):
            img = Image.open(os.path.join(DATA_DIR, file))
            coco_data["images"].append(
                {
                    "file_name": file,
                    "height": img.height,
                    "width": img.width,
                    "id": int(file[7:-4]),
                }
            )
        elif file.startswith(prefix) and file.endswith(".xml"):
            # print(file)
            tree = ET.parse(os.path.join(DATA_DIR, file))
            root = tree.getroot()
            assert len(root.findall("./table/Coords")) > 0
            for table_id in range(len(root.findall("./table/Coords"))):
                four_points = root.findall("./table/Coords")[table_id].attrib["points"]
                four_points = list(map(lambda x: x.split(","), four_points.split()))
                four_points = [[int(j) for j in i] for i in four_points]
                segmentation = [j for i in four_points for j in i]
                bbox = [
                    four_points[0][0],
                    four_points[0][1],
                    four_points[2][0] - four_points[0][0],
                    four_points[2][1] - four_points[0][1],
                ]
                coco_data["annotations"].append(
                    {
                        "segmentation": [segmentation],
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                        "image_id": int(file[7:-4]),
                        "bbox": bbox,
                        "category_id": 1,
                        "id": table_count,
                    }
                )
                table_count += 1

    with open(f"{ROOT}/{TRACK}/{SPLIT}.json", "w") as f:
        json.dump(coco_data, f)


def clean_img(DATA_DIR):
    for file in sorted(os.listdir(DATA_DIR)):
        if file.endswith(".JPG"):
            os.rename(os.path.join(DATA_DIR, file), os.path.join(DATA_DIR, file.replace(".JPG", ".jpg")))
        elif file.endswith(".TIFF"):
            img = Image.open(os.path.join(DATA_DIR, file))
            img.save(os.path.join(DATA_DIR, file.replace(".TIFF", ".jpg")))
            os.remove(os.path.join(DATA_DIR, file))
        elif file.endswith(".png"):
            img = Image.open(os.path.join(DATA_DIR, file))
            img.save(os.path.join(DATA_DIR, file.replace(".png", ".jpg")))
            os.remove(os.path.join(DATA_DIR, file))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--target_dir', required=True)
    args = parser.parse_args()

    test_data_dir = os.path.join(args.root_dir, 'test', 'TRACKA')
    test_gt_dir = os.path.join(args.root_dir, 'test_ground_truth', 'TRACKA')
    training_data_dir = os.path.join(args.root_dir, 'training', 'TRACKA', 'ground_truth')

    raw_datas = {"train": [training_data_dir], "test": [test_data_dir, test_gt_dir]}

    TRACKS = ["trackA_modern", "trackA_archival"]
    SPLITS = ["train", "test"]
    for track in TRACKS:
        prefix = "cTDaR_t0" if track == "trackA_archival" else "cTDaR_t1"
        for split in SPLITS:
            os.makedirs(os.path.join(args.target_dir, track, split))
            for source_dir in raw_datas[split]:
                for fn in os.listdir(source_dir):
                    if fn.startswith(prefix):
                        ffn = os.path.join(source_dir, fn)
                        copyfile(ffn, os.path.join(args.target_dir, track, split, fn))
            clean_img(os.path.join(args.target_dir, track, split))
            convert(args.target_dir, track, split)
