import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm
from decode_string import decode_bbox_from_caption

import json


# import util.dist as dist

#### The following loading utilities are imported from
#### https://github.com/BryanPlummer/flickr30k_entities/blob/68b3d6f12d1d710f96233f6bd2b6de799d6f4e5b/flickr30k_entities_utils.py
# Changelog:
#    - Added typing information
#    - Completed docstrings
    
def get_sentence_data(filename) -> List[Dict[str, Any]]:
    """
    Parses a sentence file from the Flickr30K Entities dataset

    input:
      filename - full file path to the sentence file to parse

    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this
                                    phrase belongs to

    """
    with open(filename, "r") as f:
        sentences = f.read().split("\n")

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == "]":
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(" ".join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == "[":
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split("/")
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {"sentence": " ".join(words), "phrases": []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data["phrases"].append(
                {"first_word_index": index, "phrase": phrase, "phrase_id": p_id, "phrase_type": p_type}
            )

        annotations.append(sentence_data)

    return annotations


def get_annotations(filename) -> Dict[str, Union[int, List[str], Dict[str, List[List[int]]]]]:
    """
    Parses the xml files in the Flickr30K Entities dataset

    input:
      filename - full file path to the annotations file to parse

    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the
                  [xmin ymin xmax ymax] format
          height - int representing the height of the image
          width - int representing the width of the image
          depth - int representing the depth of the image
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    size_container = root.findall("size")[0]
    anno_info: Dict[str, Union[int, List[str], Dict[str, List[List[int]]]]] = {}
    all_boxes: Dict[str, List[List[int]]] = {}
    all_noboxes: List[str] = []
    all_scenes: List[str] = []
    for size_element in size_container:
        assert size_element.text
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall("object"):
        for names in object_container.findall("name"):
            box_id = names.text
            assert box_id
            box_container = object_container.findall("bndbox")
            if len(box_container) > 0:
                if box_id not in all_boxes:
                    all_boxes[box_id] = []
                xmin = int(box_container[0].findall("xmin")[0].text)
                ymin = int(box_container[0].findall("ymin")[0].text)
                xmax = int(box_container[0].findall("xmax")[0].text)
                ymax = int(box_container[0].findall("ymax")[0].text)
                all_boxes[box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall("nobndbox")[0].text)
                if nobndbox > 0:
                    all_noboxes.append(box_id)

                scene = int(object_container.findall("scene")[0].text)
                if scene > 0:
                    all_scenes.append(box_id)
    anno_info["boxes"] = all_boxes
    anno_info["nobox"] = all_noboxes
    anno_info["scene"] = all_scenes

    return anno_info


#### END of import from flickr30k_entities


#### Bounding box utilities imported from torchvision and converted to numpy
def box_area(boxes: np.array) -> np.array:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        area (Tensor[N]): area for each box
    """
    assert boxes.ndim == 2 and boxes.shape[-1] == 4
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def _box_inter_union(boxes1: np.array, boxes2: np.array) -> Tuple[np.array, np.array]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union


def box_iou(boxes1: np.array, boxes2: np.array) -> np.array:
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou


#### End of import of box utilities


def _merge_boxes(boxes: List[List[int]]) -> List[List[int]]:
    """
    Return the boxes corresponding to the smallest enclosing box containing all the provided boxes
    The boxes are expected in [x1, y1, x2, y2] format
    """
    if len(boxes) == 1:
        return boxes

    np_boxes = np.asarray(boxes)

    return [[np_boxes[:, 0].min(), np_boxes[:, 1].min(), np_boxes[:, 2].max(), np_boxes[:, 3].max()]]


class RecallTracker:
    """ Utility class to track recall@k for various k, split by categories"""

    def __init__(self, topk: Sequence[int]):
        """
        Parameters:
           - topk : tuple of ints corresponding to the recalls being tracked (eg, recall@1, recall@10, ...)
        """

        self.total_byk_bycat: Dict[int, Dict[str, int]] = {k: defaultdict(int) for k in topk}
        self.positives_byk_bycat: Dict[int, Dict[str, int]] = {k: defaultdict(int) for k in topk}

    def add_positive(self, k: int, category: str):
        """Log a positive hit @k for given category"""
        if k not in self.total_byk_bycat:
            raise RuntimeError(f"{k} is not a valid recall threshold")
        self.total_byk_bycat[k][category] += 1
        self.positives_byk_bycat[k][category] += 1

    def add_negative(self, k: int, category: str):
        """Log a negative hit @k for given category"""
        if k not in self.total_byk_bycat:
            raise RuntimeError(f"{k} is not a valid recall threshold")
        self.total_byk_bycat[k][category] += 1

    def report(self) -> Dict[int, Dict[str, float]]:
        """Return a condensed report of the results as a dict of dict.
        report[k][cat] is the recall@k for the given category
        """
        report: Dict[int, Dict[str, float]] = {}
        for k in self.total_byk_bycat:
            assert k in self.positives_byk_bycat
            report[k] = {
                cat: self.positives_byk_bycat[k][cat] / self.total_byk_bycat[k][cat] for cat in self.total_byk_bycat[k]
            }
        return report


class Flickr30kEntitiesRecallEvaluator:
    def __init__(
        self,
        flickr_path: str,
        subset: str = "test",
        topk: Sequence[int] = (1, 5, 10, -1),
        iou_thresh: float = 0.5,
        merge_boxes: bool = False,
        verbose: bool = True,
    ):

        assert subset in ["train", "test", "val"], f"Wrong flickr subset {subset}"

        self.topk = topk
        self.iou_thresh = iou_thresh

        flickr_path = Path(flickr_path)

        # We load the image ids corresponding to the current subset
        with open(flickr_path / f"{subset}.txt") as file_d:
            self.img_ids = [line.strip() for line in file_d]

        if verbose:
            print(f"Flickr subset contains {len(self.img_ids)} images")

        # Read the box annotations for all the images
        self.imgid2boxes: Dict[str, Dict[str, List[List[int]]]] = {}

        if verbose:
            print("Loading annotations...")

        for img_id in self.img_ids:
            anno_info = get_annotations(flickr_path / "Annotations" / f"{img_id}.xml")["boxes"]
            if merge_boxes:
                merged = {}
                for phrase_id, boxes in anno_info.items():
                    merged[phrase_id] = _merge_boxes(boxes)
                anno_info = merged
            self.imgid2boxes[img_id] = anno_info

        # Read the sentences annotations
        self.imgid2sentences: Dict[str, List[List[Optional[Dict]]]] = {}

        if verbose:
            print("Loading annotations...")

        self.all_ids: List[str] = []
        tot_phrases = 0
        for img_id in self.img_ids:
            sentence_info = get_sentence_data(flickr_path / "Sentences" / f"{img_id}.txt")
            self.imgid2sentences[img_id] = [None for _ in range(len(sentence_info))]

            # Some phrases don't have boxes, we filter them.
            for sent_id, sentence in enumerate(sentence_info):
                phrases = [phrase for phrase in sentence["phrases"] if phrase["phrase_id"] in self.imgid2boxes[img_id]]
                if len(phrases) > 0:
                    self.imgid2sentences[img_id][sent_id] = phrases
                tot_phrases += len(phrases)

            self.all_ids += [
                f"{img_id}_{k}" for k in range(len(sentence_info)) if self.imgid2sentences[img_id][k] is not None
            ]

        if verbose:
            print(f"There are {tot_phrases} phrases in {len(self.all_ids)} sentences to evaluate")

    def evaluate(self, predictions: List[Dict]):
        evaluated_ids = set()

        recall_tracker = RecallTracker(self.topk)

        for pred in predictions:
            cur_id = f"{pred['image_id']}_{pred['sentence_id']}"
            if cur_id in evaluated_ids:
                print(
                    "Warning, multiple predictions found for sentence"
                    f"{pred['sentence_id']} in image {pred['image_id']}"
                )
                continue

            # Skip the sentences with no valid phrase
            if cur_id not in self.all_ids:
                if len(pred["boxes"]) != 0:
                    print(
                        f"Warning, in image {pred['image_id']} we were not expecting predictions "
                        f"for sentence {pred['sentence_id']}. Ignoring them."
                    )
                continue

            evaluated_ids.add(cur_id)

            pred_boxes = pred["boxes"]
            if str(pred["image_id"]) not in self.imgid2sentences:
                raise RuntimeError(f"Unknown image id {pred['image_id']}")
            if not 0 <= int(pred["sentence_id"]) < len(self.imgid2sentences[str(pred["image_id"])]):
                raise RuntimeError(f"Unknown sentence id {pred['sentence_id']}" f" in image {pred['image_id']}")
            target_sentence = self.imgid2sentences[str(pred["image_id"])][int(pred["sentence_id"])]

            phrases = self.imgid2sentences[str(pred["image_id"])][int(pred["sentence_id"])]
            if len(pred_boxes) != len(phrases):
                raise RuntimeError(
                    f"Error, got {len(pred_boxes)} predictions, expected {len(phrases)} "
                    f"for sentence {pred['sentence_id']} in image {pred['image_id']}"
                )

            for cur_boxes, phrase in zip(pred_boxes, phrases):
                target_boxes = self.imgid2boxes[str(pred["image_id"])][phrase["phrase_id"]]

                ious = box_iou(np.asarray(cur_boxes), np.asarray(target_boxes))
                for k in self.topk:
                    maxi = 0
                    if k == -1:
                        maxi = ious.max()
                    else:
                        assert k > 0
                        maxi = ious[:k].max()
                    if maxi >= self.iou_thresh:
                        recall_tracker.add_positive(k, "all")
                        for phrase_type in phrase["phrase_type"]:
                            recall_tracker.add_positive(k, phrase_type)
                    else:
                        recall_tracker.add_negative(k, "all")
                        for phrase_type in phrase["phrase_type"]:
                            recall_tracker.add_negative(k, phrase_type)

        if len(evaluated_ids) != len(self.all_ids):
            print("ERROR, the number of evaluated sentence doesn't match. Missing predictions:")
            un_processed = set(self.all_ids) - evaluated_ids
            for missing in un_processed:
                img_id, sent_id = missing.split("_")
                print(f"\t sentence {sent_id} in image {img_id}")
            raise RuntimeError("Missing predictions")

        return recall_tracker.report()

class Flickr30kEntitiesRecallEvaluatorFromTxt(Flickr30kEntitiesRecallEvaluator):
    def evaluate(self, 
                 annotation_file: str,
                 prediction_file: str,
                 quantized_size: int = 32,
                 verbose: bool = False,
                ):
        """
        annotation file is the gt json file, we use that to generate input for kosmos
        prediction file is the output from kosmos
        """
        recall_tracker = RecallTracker(self.topk)
        
        gt_json = json.load(open(annotation_file, 'r', encoding='utf-8'))
        with open(prediction_file, 'r', encoding='utf-8') as f:
            predict_all_lines = f.readlines()
        # filter the invaild lines for predict_all_lines
        filter_prediction_lines = []
        for line in predict_all_lines:
            line_pieces = line.strip('\n').split('\t')
            if 'H-' in line_pieces[0]:
                if line_pieces[0].split('-')[-1].isdigit():
                    filter_prediction_lines.append(line)
        
        predict_all_lines = filter_prediction_lines
        predict_index = -1
        
        valid_cnt = 0
        for item in tqdm(gt_json['images']):
            # file_name = item["file_name"]
            caption = item["caption"]
            img_height = float(item['height'])
            img_width = float(item['width'])
            postive_item_pos = item['tokens_positive_eval']
            # to verify 
            phrases_from_self = self.imgid2sentences[str(item['original_img_id'])][int(item['sentence_id'])]
            for pos in postive_item_pos:
                # pdb.set_trace()
                if predict_index == len(predict_all_lines):
                    break
                predict_index += 1
                
                pos_start, pos_end = pos[0]
                phrase = caption[pos_start:pos_end]
                phrase_from_self = [p for p in phrases_from_self if p['phrase'] == phrase]
                if len(phrase_from_self) == 0:
                    raise ValueError(f"Can't find the corresponding gt from two file {phrase} vs. {phrases_from_self}")
                else:
                    phrase_from_self = phrase_from_self[0]
                    
                # get the prediction from text line
                try:
                    prediction_line = predict_all_lines[predict_index]
                except IndexError as e:
                    print("Raise Indexerror.")
                    print(f"prediction index / length: {predict_index} / {len(predict_all_lines)}")
                    import sys
                    sys.exit(0)
                
                collect_entity_location = decode_bbox_from_caption(prediction_line, quantized_size=quantized_size, verbose=verbose)
                
                predict_boxes = []
                for (p_pred, p_x1, p_y1, p_x2, p_y2) in collect_entity_location:
                    if p_pred.strip() != phrase.strip(): # get the matched noun phrase
                        continue
                    else:
                        pred_box = [p_x1 * img_width, p_y1 * img_height, p_x2 * img_width, p_y2 * img_height]
                        predict_boxes.append(pred_box)
                
                if len(predict_boxes) == 0:
                    print(f"Can't find valid bbox for the given phrase ({phrase}) in caption ({caption}), \n{collect_entity_location}")
                    print(f"We set a 0-area box to calculate recall result")
                    predict_boxes = [[0., 0., 0., 0.]]
                
                # evaluate
                target_boxes = self.imgid2boxes[str(item['original_img_id'])][phrase_from_self["phrase_id"]]
                ious = box_iou(np.asarray(predict_boxes), np.asarray(target_boxes))
                for k in self.topk:
                    maxi = 0
                    if k == -1:
                        maxi = ious.max()
                    else:
                        assert k > 0
                        maxi = ious[:k].max()
                    if maxi >= self.iou_thresh:
                        recall_tracker.add_positive(k, "all")
                        for phrase_type in phrase_from_self["phrase_type"]:
                            recall_tracker.add_positive(k, phrase_type)
                    else:
                        recall_tracker.add_negative(k, "all")
                        for phrase_type in phrase_from_self["phrase_type"]:
                            recall_tracker.add_negative(k, phrase_type)
                            
                # pdb.set_trace()
                valid_cnt += 1
        print(f"Valid prediction {valid_cnt}/{len(predict_all_lines)}")     
        self.results = recall_tracker.report()
        return self.results
    
    def summarize(self):
        table = PrettyTable()
        all_cat = sorted(list(self.results.values())[0].keys())
        table.field_names = ["Recall@k"] + all_cat

        score = {}
        for k, v in self.results.items():
            cur_results = [v[cat] for cat in all_cat]
            header = "Upper_bound" if k == -1 else f"Recall@{k}"

            for cat in all_cat:
                score[f"{header}_{cat}"] = v[cat]
            table.add_row([header] + cur_results)

        print(table)
        return score

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('prediction_file', help='prediction_file')
    parser.add_argument('--annotation_file', default='/path/to/final_flickr_separateGT_test.json', help='annotation_file')
    parser.add_argument('--flickr_entities_path', default='/path/to/flickr30k_entities', help='flickr entities')
    parser.add_argument('--quantized_size', default=32, type=int)
    
    args = parser.parse_args()
    
    if '_test.json' in args.annotation_file:
        subset = "test"
    else:
        subset = "val"
    
    evaluator = Flickr30kEntitiesRecallEvaluatorFromTxt(
        flickr_path = args.flickr_entities_path,
        subset = subset,
        topk = (1, 5, 10, -1),
        iou_thresh = 0.5,
        merge_boxes = False,
        verbose = True,
    )
    
    evaluator.evaluate(args.annotation_file, args.prediction_file, args.quantized_size, verbose=False)
    evaluator.summarize()
