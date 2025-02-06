import argparse
import collections
import json
import sys
import os

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.mathscale.util import mathscale_is_equiv_proxy, is_correct as mathscale_is_correct


def majority_voting_predict(preds):
    if isinstance(preds, str):
        return preds

    preds = [pred for pred in preds if pred]
    if len(preds) == 0:
        return ""

    assert isinstance(preds, list)
    if isinstance(preds[0], list):
        tmp = []
        for pred in preds:
            tmp.append(str(sorted(pred)))
        pred = collections.Counter(tmp).most_common(1)[0][0]
        pred = eval(pred)
    elif isinstance(preds[0], str):
        pred = collections.Counter(preds).most_common(1)[0][0]
    else:
        # raise ValueError(f"Unknown type {type(preds[0])}")
        print(f"Unknown type {type(preds[0])}")
        pred = ""
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--output_file")
    parser.add_argument("--label_field", type=str, default="answer")
    args = parser.parse_args()

    data = [json.loads(line) for line in open(args.input_file)]
    for item in data:
        response = item["completion"]
        if isinstance(response, str):
            res, pred_clean, _ = mathscale_is_correct(response, item[args.label_field])
            if pred_clean is None:
                pred_clean = ""
            sc_pred = pred_clean
            sc_res = res
        elif isinstance(response, list):
            res = []
            pred_clean = []
            for resp in response:
                tmp_res, tmp_pred_clean, _ = mathscale_is_correct(resp, item[args.label_field])
                if tmp_pred_clean is None:
                    tmp_pred_clean = ""
                res.append(tmp_res)
                pred_clean.append(tmp_pred_clean)
            pred2res = {pred: r for pred, r in zip(pred_clean, res)}
            sc_pred = majority_voting_predict(pred_clean)
            sc_res = pred2res[sc_pred]
        else:
            raise ValueError(f"Unknown type of response: {type(response)}")

        item["pred"] = pred_clean
        item["sc_pred"] = sc_pred
        item["sc_res"] = sc_res
        item["res"] = res

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
        if "data_topic" in item:
            if "." in item["data_topic"]:
                item["data_topic"] = item["data_topic"].split(".")[0]
            acc_data_topic[item["data_topic"]] += int(res[0])
            cnt_data_topic[item["data_topic"]] += 1
        if any(res):
            pass_at_k += 1
        if item["sc_res"]:
            sc += 1

    assert pass_at_k <= len(data)
    json.dump(data, open(args.output_file, "w"), indent=2)

    if len(data) == 0:
        metrics = {"acc": 0, "pass@k": 0, "maj@k": 0, "correct": 0, "total": 0}
    else:
        metrics = {"acc": cnt / len(data), "pass@k": pass_at_k / len(data), "maj@k": sc / len(data),
                   "correct": cnt, "total": len(data)}
        if len(acc_data_topic) > 0:
            for key in acc_data_topic:
                metrics[f"acc_{key}"] = acc_data_topic[key] / cnt_data_topic[key]
    json.dump(metrics, open(args.output_file.replace(".json", ".metrics.json"), "w"), indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
