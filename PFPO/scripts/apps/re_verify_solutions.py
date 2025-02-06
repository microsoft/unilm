import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    data = json.load(open(args.input_file))

    success = 0
    success_at_k = 0
    for item in data:
        if isinstance(item["res"], bool):
            assert isinstance(item["response"], str)
            if item["pred"] is None:
                item["res"] = False

            if item["res"]:
                success += 1
                success_at_k += 1
        else:
            if len(item["res"]):
                assert isinstance(item["response"], list)
                for i in range(len(item["response"])):
                    if item["pred"][i] is None:
                        item["res"][i] = False
                        item["full_res"][i] = [[False] * len(item["test_cases"]["inputs"]) if item["test_cases"] else 1]

                if any(item["res"]):
                    success_at_k += 1
                if item["res"][0]:
                    success += 1

    metrics = {"acc": success / len(data), "pass@k": success_at_k / len(data), "correct": success, "total": len(data)}
    print(json.dumps(metrics, indent=2))
    if args.output_file is not None:
        json.dump(data, open(args.output_file, "w"), ensure_ascii=False)
        json.dump(metrics, open(args.output_file.replace(".json", ".metrics"), "w"), ensure_ascii=False)


if __name__ == '__main__':
    main()
