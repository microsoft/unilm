import json
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    data = [json.loads(line) for line in open(args.input_file).readlines()]
    outputs = []

    for i, item in tqdm(enumerate(data)):
        prompt = item["prompt"]

        _s = prompt.index("### Instruction:") + len("### Instruction:")
        _e = prompt.index("### Response:")
        question = prompt[_s:_e].strip()

        response = item["completion"]

        outputs.append({
            "id": i,
            "question": question,
            "response": response,
        })

    json.dump(outputs, open(args.output_file, "w"))


if __name__ == "__main__":
    main()
