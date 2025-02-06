import collections
import json
import argparse
import json
import os.path
from glob import glob
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.deepseek_math_utils import eval_script, answer_extraction
from post_processors.openai_api_callback import majority_voting_predict


def pred2str(pred):
    if isinstance(pred, str):
        return pred
    if isinstance(pred, list):
        pred = sorted(pred)
        pred = str(pred)
        return pred
    raise ValueError(f"Unknown type {type(pred)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--output_file")
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        data = json.load(open(args.input_file))
    else:
        data = []
        for file in glob(args.input_file):
            print(file)
            data += json.load(open(file))
        if len(data) == 0:
            raise ValueError(f"No data found in {args.input_file}")

    cnt = 0
    pass_at_k = 0
    sc = 0
    for item in data:
        if isinstance(item["response"], list):
            preds = item["pred"]
        else:
            preds = [item["pred"]]

        if "res" not in item:
            mul_pass = 0
            if len(preds) > 0:
                res = []
                for pred in preds:
                    res.append(eval_script.eval_math({"prediction": pred, "answer": item["label"]}))
                if any(res):
                    mul_pass = 1
            else:
                res = []

            item["pass_at_k"] = mul_pass

            if len(preds) == 1:
                res = res[0]
            item["res"] = res

        if not isinstance(item["res"], list):
            res = [item["res"]]
        else:
            res = item["res"]

        if any(res):
            pass_at_k += 1
        if res[0]:
            cnt += 1

        # str_preds = [pred2str(item) for item in preds]
        # counter = collections.Counter(str_preds)
        # sc_pred = eval(counter.most_common(1)[0][0])
        # sc_res = eval_script.eval_math({"prediction": sc_pred, "answer": item["label"]})

        # if "sc_res" in item:
        #     sc_res = item["sc_res"]
        # else:
        #     preds = [x for x in preds if x]
        #     if len(preds):
        #         sc_pred = majority_voting_predict(preds)
        #         try:
        #             sc_res = eval_script.eval_math({"prediction": sc_pred, "answer": item["label"]})
        #         except Exception as e:
        #             print(f"Error in {item['id']} during evaluation: {e}")
        #             sc_res = False
        #     else:
        #         sc_res = False
        # if sc_res:
        #     sc += 1

    print(f"Pass at k: {pass_at_k}/{len(data)} = {pass_at_k / len(data)}")
    print(f"Correct at k: {cnt}/{len(data)} = {cnt / len(data)}")
    print(f"Self-consistency: {sc}/{len(data)} = {sc / len(data)}")
    if args.output_file:
        json.dump(data, open(args.output_file, "w"), indent=2)


if __name__ == '__main__':
    main()
