import json
import argparse

x_code_eval_template = """{description}

Input: {input_from}
Output: {output_to}

Time limit: {time_limit}
Memory limit: {memory_limit}

-----Input-----

{input_spec}

-----Output-----

{output_spec}

-----Notes-----

{notes}

-----Example-----

{input_output}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--output_file")
    parser.add_argument("--prompt_file", type=str, default="prompts/apps/test_input_gen_2shot_v2.1.txt")
    args = parser.parse_args()

    in_out_template = "Input\n```{}```\n\nOutput\n```{}```"
    prompt_tem = open(args.prompt_file).read()

    data = [json.loads(line) for line in open(args.input_file).readlines()]
    outputs = []
    for item in data:
        input_output = []
        for _in, _out in zip(item["sample_inputs"], item["sample_outputs"]):
            input_output.append(in_out_template.format(_in, _out))

        item["input_output"] = "\n\n".join(input_output)

        question = x_code_eval_template.format(**item)

        prompt = prompt_tem.replace("[[Question]]", question)

        item["prompt"] = prompt
        outputs.append(item)

    with open(args.output_file, "w") as f:
        for item in outputs:
            f.write(json.dumps(item) + "\n")

    print(len(outputs))


if __name__ == '__main__':
    main()
