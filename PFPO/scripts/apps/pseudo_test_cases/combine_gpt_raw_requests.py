import json
import argparse
import os
import sys
import os
from tqdm import tqdm

sys.set_int_max_str_digits(0)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from post_processors.code.clean import standard_cleaner_default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--raw_output_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    input_data = [json.loads(line) for line in open(args.input_file, encoding="utf-8").readlines()]

    raw_outputs = [json.loads(line) for line in open(args.raw_output_file, encoding="utf-8").readlines()]
    for o in tqdm(raw_outputs):
        request_meta = o["request_metadata"]
        s_id = int(request_meta["sid"].split("@")[1])
        input_item = input_data[s_id]
        assert o["request"]["messages"][1]["content"] in input_item["prompt"]

        if "response" not in input_item:
            input_item["response"] = []
        if "content" not in o["response"]["choices"][0]["message"]:
            print(o)
            continue
        input_item["response"].append(o['response']['choices'][0]['message']['content'])

    missing_completion = 0
    num_completions = 0
    for item in input_data:
        if "response" not in item:
            missing_completion += 1
            item["response"] = ""
            item["pred"] = None
        else:
            preds = [standard_cleaner_default(resp) for resp in item["response"]]
            item["pred"] = preds
            num_completions += len(preds)

    json.dump(input_data, open(args.output_file, "w", encoding="utf-8"))
    print(f"Averaged number of completions: {num_completions / len(input_data)}")


if __name__ == '__main__':
    main()
