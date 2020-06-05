import torch, os, sys, re
from layoutlm.layoutlm import LayoutlmConfig, LayoutlmForSequenceClassification
from transformers import BertTokenizerFast
from layoutlm.layoutlm.data.rvl_cdip import CdipProcessor, get_prop, DocExample, convert_examples_to_features
from lxml import html


# from rvl_cdip.py
def read_hocr_file(hocr_file):
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
    return text_buffer, bbox_buffer


def modelPredict(output_path, hocr_file):
    # config, tokenizer, and model all loaded from output directory
    config = LayoutlmConfig.from_pretrained(output_path)
    tokenizer = BertTokenizerFast.from_pretrained(output_path)
    model = LayoutlmForSequenceClassification.from_pretrained(output_path, config=config)

    processor = CdipProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    text, bbox = read_hocr_file(hocr_file)
    # next step from here in run_classification.py is to create DocExample, convert to DocFeature, then to a Tensor
    # that we can properly pass into the bottom call. this is the part that I have been stuck on, as the conversion
    # outlined in run_classification.py requires labels and other inputs that aren't directly obvious what they should be
    # when it comes to us trying to predict on a new image.
    model("OUR TENSOR CONTAINING THE IMAGE INFO SHOULD GO HERE")

if __name__ == "__main__":
    # modelPredict(sys.argv[1])
    modelPredict("/Users/chris/CODE/cedrus/unilm/layoutlm/examples/classification/aetna_dataset_output_base_20_d1"
                 , "/Users/chris/CODE/cedrus/unilm/layoutlm/layoutlm/data/Aetna Dataset -2/images/test/COB1-1.xml")