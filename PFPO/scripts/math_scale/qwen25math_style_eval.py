import argparse
import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pebble import ProcessPool
from functools import partial
from multiprocessing.pool import Pool
import re

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.qwen25math.grader import math_equal
from data.qwen25math.parser import extract_answer, strip_string, STRIP_EXCEPTIONS


def extract_content_from_tag(pred: str):
    # Regular expression pattern to match the content between <answer> and </answer>
    pattern = r'<answer>(.*?)</answer>'
    # Use re.DOTALL to allow matching newlines within the tags
    match = re.search(pattern, pred, re.DOTALL)
    if match:
        return match.group(1).strip()  # Strip removes extra spaces or newlines
    return pred


def majority_voting_predict(preds):
    if isinstance(preds, str):
        return preds

    preds = [pred for pred in preds if pred]
    if len(preds) == 0:
        return "", 0

    assert isinstance(preds, list)
    if isinstance(preds[0], list):
        tmp = []
        for pred in preds:
            tmp.append(str(sorted(pred)))
        pred, freq = collections.Counter(tmp).most_common(1)[0]
        pred = eval(pred)
    elif isinstance(preds[0], str):
        pred, freq = collections.Counter(preds).most_common(1)[0]
    else:
        # raise ValueError(f"Unknown type {type(preds[0])}")
        print(f"Unknown type {type(preds[0])}")
        pred = ""
        freq = 0

    freq = freq / len(preds)

    return pred, freq


def _annotate(param):
    return param[0], math_equal(param[-2], param[-1])


def preprocess_item(item, args):
    response = item[args.response_field]
    if isinstance(response, str):
        response = extract_content_from_tag(response)
        pred_clean = extract_answer(response, data_name="math")
        pred_clean = strip_string(pred_clean, skip_unit="math" in STRIP_EXCEPTIONS)
        if pred_clean is None:
            pred_clean = ""
        sc_pred = pred_clean
        sc_freq = 1.0
    elif isinstance(response, list):
        pred_clean = []
        for resp in response:
            resp = extract_content_from_tag(resp)
            tmp_pred_clean = extract_answer(resp, data_name="math")
            tmp_pred_clean = strip_string(tmp_pred_clean, skip_unit="math" in STRIP_EXCEPTIONS)
            if tmp_pred_clean is None:
                tmp_pred_clean = ""
            pred_clean.append(tmp_pred_clean)
        sc_pred, sc_freq = majority_voting_predict(pred_clean)
    else:
        raise ValueError(f"Unknown type of response: {type(response)}")
    item["pred"] = pred_clean
    item["sc_pred"] = sc_pred
    item["sc_freq"] = sc_freq

    if not isinstance(item["pred"], list):
        preds = [item["pred"]]
    else:
        preds = item["pred"]

    if "college_math" in item[args.source_field]:
        item[args.label_field] = item[args.label_field].replace("$", "").strip()

    data_name = item[args.source_field].split(".")[0]
    if data_name not in STRIP_EXCEPTIONS:
        item[args.label_field] = strip_string(item[args.label_field], skip_unit=data_name == "carp_en")
    else:
        # gt_ans = (
        #     gt_ans.replace("\\neq", "\\ne")
        #     .replace("\\leq", "\\le")
        #     .replace("\\geq", "\\ge")
        # )
        raise NotImplementedError()

    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--sub_category", type=str, default=None)
    parser.add_argument("--label_field", type=str, default="label")
    parser.add_argument("--response_field", type=str, default="response")
    parser.add_argument("--source_field", type=str, default="data_topic")
    args = parser.parse_args()

    if args.input_file.endswith(".json"):
        data = json.load(open(args.input_file))
    else:
        data = [json.loads(line) for line in open(args.input_file).readlines()]

    if args.sub_category is not None:
        print(args.sub_category)
        sub_categories = set(list(args.sub_category.split(",")))
        data = [item for item in data if any([sub_category in item[args.source_field] for sub_category in sub_categories])]

    _mp_inputs = []
    with Pool(args.num_workers) as p:
        results = list(tqdm(p.imap(partial(preprocess_item, args=args), data), total=len(data), desc="Preprocess data"))

    for i, item in tqdm(enumerate(results), total=len(results), desc="Preprocess data"):
        if not isinstance(item["pred"], list):
            preds = [item["pred"]]
        else:
            preds = item["pred"]
        for j, pred in enumerate(preds):
            _mp_inputs.append(((i, j), pred, str(item[args.label_field])))

    pbar = tqdm(_mp_inputs, total=len(_mp_inputs), desc="Submitting eval task", dynamic_ncols=True)

    outputs = collections.defaultdict(dict)
    timeout_cnt = 0

    with ProcessPool(max_workers=1) as pool:
        future = pool.map(_annotate, pbar, timeout=3)
        iterator = future.result()
        with tqdm(total=len(_mp_inputs), desc="Evaluate") as progress_bar:
            while True:
                try:
                    idx, result = next(iterator)
                    # scores.append(result)
                    outputs[idx[0]][idx[1]] = result
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    # outputs[idx[0]][idx[1]] = False
                    timeout_cnt += 1
                except Exception as error:
                    print(error)
                    # exit()
                progress_bar.update(1)

    for i, item in enumerate(data):
        if not isinstance(item["pred"], list):
            preds = [item["pred"]]
        else:
            preds = item["pred"]
        if i not in outputs:
            all_res = [False] * len(preds)
        else:
            all_res = outputs[i]

        for j, pred in enumerate(preds):
            if j not in all_res:
                all_res[j] = False

        assert len(all_res) == len(preds)
        pred2res = {pred: all_res[j] for j, pred in enumerate(preds)}
        sc_res = pred2res[item["sc_pred"]]

        item["res"] = [pred2res[pred] for pred in preds]
        item["sc_res"] = sc_res
        assert "sc_freq" in item

        if not isinstance(item["pred"], list):
            assert len(item["res"]) == 1
            item["res"] = item["res"][0]

    cnt = 0
    pass_at_k = 0
    sc = 0
    acc_data_topic = collections.Counter()
    cnt_data_topic = collections.Counter()
    for item in data:
        if not isinstance(item["res"], list):
            res = [item["res"]]
        else:
            res = item["res"]
        if res[0]:
            cnt += 1
        if args.source_field in item:
            if "." in item[args.source_field]:
                item[args.source_field] = item[args.source_field].split(".")[0]
            acc_data_topic[item[args.source_field]] += int(res[0])
            cnt_data_topic[item[args.source_field]] += 1
        if any(res):
            pass_at_k += 1
        if item["sc_res"]:
            sc += 1

    output_file = args.input_file.replace(".json", ".sympy_eval.json")
    assert pass_at_k <= len(data)
    json.dump(data, open(output_file, "w"), indent=2)

    if len(data) == 0:
        metrics = {"acc": 0, "pass@k": 0, "maj@k": 0, "correct": 0, "total": 0}
    else:
        metrics = {"acc": cnt / len(data), "pass@k": pass_at_k / len(data), "maj@k": sc / len(data),
                   "correct": cnt, "total": len(data)}
        if len(acc_data_topic) > 0:
            for key in acc_data_topic:
                metrics[f"acc_{key}"] = acc_data_topic[key] / cnt_data_topic[key]
    json.dump(metrics, open(output_file.replace(".json", ".metrics.json"), "w"), indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
