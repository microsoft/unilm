import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--prompt_file", type=str, default="prompts/apps/magicoder_cls_2shot.txt")
    args = parser.parse_args()

    data = [json.loads(line) for line in open(args.input_file, encoding='utf-8')]
    prompt_template = open(args.prompt_file, 'r').read()

    for i, item in enumerate(data):
        prompt = prompt_template.replace("[[Question]]", item["instruction"])
        prompt = prompt.replace("[[Solution]]", item["response"])
        data[i]["prompt"] = prompt
        data[i]["id"] = i

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
