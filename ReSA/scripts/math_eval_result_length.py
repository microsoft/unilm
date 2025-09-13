import argparse
from transformers import LlamaTokenizerFast
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from math_utils import evaluate, load_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k", type=str)
    parser.add_argument("--result_file", default=None, type=str)
    parser.add_argument("--prompt_type", default="direct", type=str)
    parser.add_argument("--eval_num", default=-1, type=int)
    args = parser.parse_args()
    return args


def eval_math_acc(args, data_name):
    result_file = args.result_file.format(data_name=data_name)
    print(result_file)
    all_samples = list(load_jsonl(result_file))
    if args.eval_num > 0:
        all_samples = all_samples[: args.eval_num]
    
    tokenizer = LlamaTokenizerFast.from_pretrained('/mnt/msranlp/tianzhu/ckpt/DeepSeek-R1-Distill-Qwen-1.5B')
    
    avg_len = 0
    threshold = 2048
    below_num, above_num, below_acc, above_acc = 0, 0, 0, 0
    for sample in all_samples:
        length = len(tokenizer.encode(sample["code"][0]))
        avg_len += length
        _, result_json = evaluate(
            samples=[sample],
            data_name=data_name,
            prompt_type=args.prompt_type,
            execute=True,
        )
        if length <= threshold:
            below_num += 1
            below_acc += result_json["acc"]
        else:
            above_num += 1
            above_acc += result_json["acc"]
    total_num = below_num + above_num
    total_acc = (below_acc + above_acc) / total_num
    avg_len /= len(all_samples)
    print(f"{data_name} total acc: {total_acc:.1f} ({total_num})")
    print(
        f"{data_name} below {threshold} acc: {int(below_acc/100)}/{below_num}/{below_acc/below_num if below_num > 0 else 0:.1f}"
    )
    print(
        f"{data_name} above {threshold} acc: {int(above_acc/100)}/{above_num}/{above_acc/above_num if above_num > 0 else 0:.1f}"
    )
    print(f"{data_name} avg len: {avg_len:.1f}")
    
    print(f"{total_acc:.1f}")
    print(f"{int(below_acc/100)}/{below_num}/{below_acc/below_num if below_num > 0 else 0:.1f}")
    print(f"{int(above_acc/100)}/{above_num}/{above_acc/above_num if above_num > 0 else 0:.1f}")
    print(f"{avg_len:.1f}")
    # print(result_json)

    return result_json


def main(args):
    data_names = args.data_names
    data_list = data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(eval_math_acc(args, data_name))

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


if __name__ == "__main__":
    args = parse_args()
    main(args)