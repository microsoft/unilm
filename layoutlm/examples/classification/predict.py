import os
import re
import sys
import torch
from lxml import html
from transformers import BertTokenizerFast

from layoutlm.modeling.layoutlm import LayoutlmConfig, LayoutlmForSequenceClassification
from layoutlm.data.rvl_cdip import CdipProcessor, get_prop, DocExample, convert_examples_to_features


# from rvl_cdip.py
def convert_hocr_to_feature(hocr_file, tokenizer, label_list, label):
    text_buffer = []
    bbox_buffer = []
    try:
        doc = html.parse(hocr_file)
    except AssertionError:
        return [], []
    for page in doc.xpath("//*[@class='ocr_page']"):
        page_bbox = [int(x) for x in get_prop(page, "bbox").split()]
        width, height = page_bbox[2], page_bbox[3]
        for word in doc.xpath("//*[@class='ocrx_word']"):
            textnodes = word.xpath(".//text()")
            s = "".join([text for text in textnodes])
            text = re.sub(r"\s+", " ", s).strip()
            if text:
                text_buffer.append(text)
                bbox = [int(x) for x in get_prop(word, "bbox").split()]
                bbox = [
                    bbox[0] / width,
                    bbox[1] / height,
                    bbox[2] / width,
                    bbox[3] / height,
                ]
                bbox = [int(x * 1000) for x in bbox]
                bbox_buffer.append(bbox)
    # hocr file is now read, all relevant data is in text_buffer and bbox_buffer
    guid = "eval-0"
    # convert from hocr data to DocExample
    examples = [DocExample(guid=guid, text_a=text_buffer, text_b=None, bbox=bbox_buffer, label=label)]
    # convert from DocExample to list of DocFeature
    features = convert_examples_to_features(examples=examples, tokenizer=tokenizer, label_list=label_list)
    return features[0]


def make_prediction(output_path, hocr_file):
    # config, tokenizer, and model all loaded from output directory
    config = LayoutlmConfig.from_pretrained(output_path)
    tokenizer = BertTokenizerFast.from_pretrained(output_path)
    model = LayoutlmForSequenceClassification.from_pretrained(output_path, config=config)

    processor = CdipProcessor()
    label_list = processor.get_labels()
    feature = convert_hocr_to_feature(hocr_file, tokenizer, label_list, "0")

    model.eval()
    with torch.no_grad():
        inputs = {
            "input_ids": torch.tensor([feature.input_ids]),
            "attention_mask": torch.tensor([feature.attention_mask]),
            "token_type_ids": torch.tensor([feature.token_type_ids]),
            "labels": torch.tensor([feature.label]),
            "bbox": torch.tensor([feature.bboxes])
        }
        outputs = model(**inputs)
        sm = torch.nn.Softmax()
        probabilities = sm(outputs[1]).tolist()[0]
        max_prob, max_index, index = 0, 0, 0
        for p in probabilities:
            if p > max_prob:
                max_prob = p
                max_index = index
            index += 1
        # returns index of label for now, we can likely reference label_list in the future to return label name???
    return max_index, max_prob


if __name__ == "__main__":
    label, confidence = make_prediction('/Users/chris/CODE/cedrus/unilm/layoutlm/examples/classification/aetna_dataset_output_base_40_d3',
                    '/Users/chris/CODE/cedrus/unilm/layoutlm/layoutlm/data/Aetna Dataset -3/OCR/images/COB1/COB1-1.xml')
    print('>>> Predicted label %s with %s%% confidence' % (label, confidence * 100))
    #label, confidence = make_prediction(sys.argv[1], sys.argv[2])
    #head, tail = os.path.split(sys.argv[2])
    #print('>>> Predicted label %s with %s%% confidence for input file %s' % (label, confidence*100, tail))