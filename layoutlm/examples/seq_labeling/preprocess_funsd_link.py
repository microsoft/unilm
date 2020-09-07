import argparse
import json
import os
from glob import glob
import pdb
import uuid

from PIL import Image
from transformers import AutoTokenizer

"""
python preprocess_funsd_link.py --data_dir funsd/dataset/training_data/annotations --data_split train --model_name_or_path bert-base-uncased --max_len 510 --output_dir funsd/dataset
"""


def bbox_string(box, width, length):
    return (
        str(int(1000 * (box[0] / width)))
        + " "
        + str(int(1000 * (box[1] / length)))
        + " "
        + str(int(1000 * (box[2] / width)))
        + " "
        + str(int(1000 * (box[3] / length)))
    )


def actual_bbox_string(box, width, length):
    return (
        str(box[0])
        + " "
        + str(box[1])
        + " "
        + str(box[2])
        + " "
        + str(box[3])
        + "\t"
        + str(width)
        + " "
        + str(length)
    )


"""
qas_id: 5733be284776f41900661182,
question_text: To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?,
doc_tokens: [Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.],
start_position: 90,
end_position: 92,
orig_answer_text: Saint Bernadette Soubirous,
is_impossible: False
"""


def convert(args):
    out = dict()
    fnames = glob(args.data_dir + '/*.json')
    for fn in fnames:
        print(fn)
        with open(fn, "r", encoding="utf8") as f:
            data = json.load(f)
        image_path = fn.replace("annotations", "images").replace("json", "png")
        image = Image.open(image_path)
        width, length = image.size
        # id as key
        meta_cells, idx_token_map, cnt = dict(), dict(), 0
        for item in data["form"]:
            meta_cells[item["id"]] = item
            words = [w for w in item["words"] if w["text"].strip() != ""]
            if len(words) == 0:
                continue
            idx_token_map[item["id"]] = (cnt, cnt + len(words) - 1)
            cnt += len(words)

        example = dict()
        token_idx = 0
        # one image
        doc_tokens, bboxes, actual_bboxes, qas = [], [], [], []
        for item in data["form"]:
            words, label, linking, idx = item["words"], item["label"], item["linking"], item["id"]
            words = [w for w in words if w["text"].strip() != ""]
            if len(words) == 0:
                continue
            for w in words:
                doc_tokens.append(w['text'].strip())
                bboxes.append(bbox_string(w["box"], width, length))
                actual_bboxes.append(
                    actual_bbox_string(w["box"], width, length))
            if label == 'question' and len(linking) == 1:
                q, a = linking[0][:]
                if q != idx:
                    q_cell = meta_cells[q]
                    # print(
                    #    'not valid qustion with q/a {}/{}'.format(q_cell['text'], a_cell['text']))
                else:
                    a_cell = meta_cells[a]
                    if a_cell['label'] == 'answer' and len(a_cell['text']) > 0:
                        # print(
                        #    'valid qa pair {}/{}'.format(item['text'], a_cell['text']))
                        question_start_position = token_idx
                        question_end_position = token_idx + len(words) - 1
                        question_text = item['text']
                        orig_answer_text = a_cell['text']
                        answer_start_position, answer_end_position = idx_token_map[a_cell['id']]
                        uid = str(uuid.uuid1())
                        qas.append(dict(question_text=question_text,
                                        answer_start_position=answer_start_position,
                                        answer_end_position=answer_end_position,
                                        question_start_position=question_start_position,
                                        question_end_position=question_end_position,
                                        orig_answer_text=orig_answer_text,
                                        uid=uid))

            token_idx += len(words)
        for qa in qas:
            orig_answer_text = qa['orig_answer_text']
            question_text = qa['question_text']
            q_tokens = doc_tokens[qa['question_start_position']
                : qa['question_end_position'] + 1]
            a_tokens = doc_tokens[qa['answer_start_position']
                : qa['answer_end_position'] + 1]
            print("ori k/v {} / {}".format(question_text, orig_answer_text))
            print("k/v {} / {}".format(' '.join(q_tokens), ' '.join(a_tokens)))
        # pdb.set_trace()
        example.update(doc_tokens=doc_tokens, bboxes=bboxes,
                       actual_bboxes=actual_bboxes, file_name=fn, qas=qas)
        out[fn] = example
    with open(os.path.join(args.output_dir, args.data_split + '.json'), 'w') as f:
        json.dump(out, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="data/training_data/annotations"
    )
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--model_name_or_path", type=str,
                        default="bert-base-uncased")
    parser.add_argument("--max_len", type=int, default=510)
    args = parser.parse_args()

    convert(args)
