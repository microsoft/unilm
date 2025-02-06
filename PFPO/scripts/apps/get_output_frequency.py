import copy
import json
import argparse
import collections
import sys
from tqdm import tqdm

sys.set_int_max_str_digits(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    args = parser.parse_args()

    data = json.load(open(args.input_file))

    missed = 0
    averaged_rank = 0
    case_cnt = 0

    loose_avg_rank = 0
    loose_missed = 0
    loose_case_cnt = 0

    tmp_cnt = 0
    for item in tqdm(data):
        if "outputs" not in item:
            print("No outputs")
            continue

        test_cases = item["input_output"]
        if not test_cases:
            print("No ground truth test cases")
            continue

        test_case_outputs = collections.defaultdict(collections.Counter)
        case_level_res = collections.defaultdict(dict)
        for i, sol_outputs in enumerate(item["outputs"]):
            for j, o in enumerate(sol_outputs):
                if o is not None:
                    test_case_outputs[j][o] += 1
                    if len(item["full_res"][i]) < len(sol_outputs):
                        # print("Full res is not complete")
                        tmp_cnt += 1
                        continue
                    case_level_res[j][o] = item["full_res"][i][j]

        for i, o in enumerate(test_cases["outputs"]):
            o = str(o)
            cnt = test_case_outputs[i]
            sorted_cnt = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
            o2rank = {k: i for i, (k, v) in enumerate(sorted_cnt)}

            if o not in o2rank:
                missed += 1
            else:
                rank = o2rank[o]
                averaged_rank += rank
                case_cnt += 1

            pseudo_cnt = 0
            loose_cnt = copy.deepcopy(cnt)
            loose_tuples = copy.deepcopy(list(loose_cnt.items()))
            for real_o, real_o_res in loose_tuples:
                if real_o_res:
                    pseudo_cnt += test_case_outputs[i][real_o]
                    loose_cnt.pop(real_o)

            if pseudo_cnt == 0:
                loose_missed += 1
                continue

            loose_cnt["$$LOOSE_POS$$"] = pseudo_cnt
            loose_sorted_cnt = sorted(loose_cnt.items(), key=lambda x: x[1], reverse=True)
            loose_o2rank = {k: i for i, (k, v) in enumerate(loose_sorted_cnt)}
            loose_rank = loose_o2rank.get("$$LOOSE_POS$$", -1)
            assert loose_rank != -1
            loose_avg_rank += loose_rank
            loose_case_cnt += 1

    print(f"Missed: {missed}")
    print(f"Averaged rank: {averaged_rank / case_cnt}")
    print(f"Case count: {case_cnt}")
    print(f"Total count: {len(data)}")

    print(f"Loose missed: {loose_missed}")
    print(f"Loose averaged rank: {loose_avg_rank / loose_case_cnt}")
    print(f"Loose case count: {loose_case_cnt}")
    print(f"Loose total count: {len(data)}")

    print(f"tmp_cnt: {tmp_cnt}")


if __name__ == "__main__":
    main()


"""
>>> python scripts/apps/get_output_frequency.py --input_file outputs/apps/apps.train.r2c.vanilla.gpt-4o.tem1.0.n11.exec.w_outputs.json

Missed: 24502
Averaged rank: 0.20820668693009117
Case count: 1316
Total count: 5000
Loose missed: 8206
Loose averaged rank: 0.0
Loose case count: 17612
Loose total count: 5000
tmp_cnt: 3077
"""

