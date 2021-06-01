import argparse
import json
import os

from PIL import Image
from transformers import AutoTokenizer


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


def convert(args):
    with open(
        os.path.join(args.output_dir, args.data_split + ".txt.tmp"),
        "w",
        encoding="utf8",
    ) as fw, open(
        os.path.join(args.output_dir, args.data_split + "_box.txt.tmp"),
        "w",
        encoding="utf8",
    ) as fbw, open(
        os.path.join(args.output_dir, args.data_split + "_image.txt.tmp"),
        "w",
        encoding="utf8",
    ) as fiw:
        for file in os.listdir(args.data_dir):
            file_path = os.path.join(args.data_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = file_path.replace("annotations", "images")
            image_path = image_path.replace("json", "png")
            file_name = os.path.basename(image_path)
            image = Image.open(image_path)
            width, length = image.size
            for item in data["form"]:
                words, label = item["words"], item["label"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                if label == "other":
                    for w in words:
                        fw.write(w["text"] + "\tO\n")
                        fbw.write(
                            w["text"]
                            + "\t"
                            + bbox_string(w["box"], width, length)
                            + "\n"
                        )
                        fiw.write(
                            w["text"]
                            + "\t"
                            + actual_bbox_string(w["box"], width, length)
                            + "\t"
                            + file_name
                            + "\n"
                        )
                else:
                    if len(words) == 1:
                        fw.write(words[0]["text"] + "\tS-" + label.upper() + "\n")
                        fbw.write(
                            words[0]["text"]
                            + "\t"
                            + bbox_string(words[0]["box"], width, length)
                            + "\n"
                        )
                        fiw.write(
                            words[0]["text"]
                            + "\t"
                            + actual_bbox_string(words[0]["box"], width, length)
                            + "\t"
                            + file_name
                            + "\n"
                        )
                    else:
                        fw.write(words[0]["text"] + "\tB-" + label.upper() + "\n")
                        fbw.write(
                            words[0]["text"]
                            + "\t"
                            + bbox_string(words[0]["box"], width, length)
                            + "\n"
                        )
                        fiw.write(
                            words[0]["text"]
                            + "\t"
                            + actual_bbox_string(words[0]["box"], width, length)
                            + "\t"
                            + file_name
                            + "\n"
                        )
                        for w in words[1:-1]:
                            fw.write(w["text"] + "\tI-" + label.upper() + "\n")
                            fbw.write(
                                w["text"]
                                + "\t"
                                + bbox_string(w["box"], width, length)
                                + "\n"
                            )
                            fiw.write(
                                w["text"]
                                + "\t"
                                + actual_bbox_string(w["box"], width, length)
                                + "\t"
                                + file_name
                                + "\n"
                            )
                        fw.write(words[-1]["text"] + "\tE-" + label.upper() + "\n")
                        fbw.write(
                            words[-1]["text"]
                            + "\t"
                            + bbox_string(words[-1]["box"], width, length)
                            + "\n"
                        )
                        fiw.write(
                            words[-1]["text"]
                            + "\t"
                            + actual_bbox_string(words[-1]["box"], width, length)
                            + "\t"
                            + file_name
                            + "\n"
                        )
            fw.write("\n")
            fbw.write("\n")
            fiw.write("\n")


def seg_file(file_path, tokenizer, max_len):
    subword_len_counter = 0
    output_path = file_path[:-4]
    with open(file_path, "r", encoding="utf8") as f_p, open(
        output_path, "w", encoding="utf8"
    ) as fw_p:
        for line in f_p:
            line = line.rstrip()

            if not line:
                fw_p.write(line + "\n")
                subword_len_counter = 0
                continue
            token = line.split("\t")[0]

            current_subwords_len = len(tokenizer.tokenize(token))

            # Token contains strange control characters like \x96 or \x95
            # Just filter out the complete line
            if current_subwords_len == 0:
                continue

            if (subword_len_counter + current_subwords_len) > max_len:
                fw_p.write("\n" + line + "\n")
                subword_len_counter = current_subwords_len
                continue

            subword_len_counter += current_subwords_len

            fw_p.write(line + "\n")


def seg(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, do_lower_case=True
    )
    seg_file(
        os.path.join(args.output_dir, args.data_split + ".txt.tmp"),
        tokenizer,
        args.max_len,
    )
    seg_file(
        os.path.join(args.output_dir, args.data_split + "_box.txt.tmp"),
        tokenizer,
        args.max_len,
    )
    seg_file(
        os.path.join(args.output_dir, args.data_split + "_image.txt.tmp"),
        tokenizer,
        args.max_len,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="data/training_data/annotations"
    )
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--max_len", type=int, default=510)
    args = parser.parse_args()

    convert(args)
    seg(args)
