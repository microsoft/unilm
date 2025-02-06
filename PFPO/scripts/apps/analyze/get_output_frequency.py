import collections
import json
import os
import sys
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from glob import glob

from tqdm import tqdm

sys.set_int_max_str_digits(0)


def worker(x):
    item, orig_item = x
    input_output = item["test_cases"]
    inputs = input_output["inputs"]

    num_test_cases = len(input_output["inputs"])

    outputs_counter = [
        collections.Counter() for _ in inputs
    ]
    output_str2orig_pred = [
        {} for _ in inputs
    ]
    resp2outputs = [
        {} for _ in range(len(item["outputs"]))
    ]

    output2res = [
        {} for _ in range(len(inputs))
    ]

    assert len(item["full_res"]) == len(item["outputs"]) == len(item["pred"]), (len(item["full_res"]), len(item["outputs"]), len(item["pred"]))
    for resp_id, (full_res, pg_outputs) in enumerate(zip(item["full_res"], item["outputs"])):
        for case_j, (case_r, case_o) in enumerate(zip(full_res, pg_outputs)):
            if case_j >= len(inputs):
                break
            if case_r != 0:
                continue
            # assert case_o  # sometimes is could be `int` or `True` or `False`. We believe the `case_r` here.

            # if not str(case_o):
            #     continue  # some outputs could be empty string

            if str(case_o) not in output_str2orig_pred[case_j]:
                output_str2orig_pred[case_j][str(case_o)] = case_o
            outputs_counter[case_j][str(case_o)] += 1
            resp2outputs[resp_id][case_j] = str(case_o)

            # assert case_j < len(orig_item["full_res"][resp_id]), (resp_id, case_j, orig_item["full_res"][resp_id])
            if len(orig_item["full_res"][resp_id]) <= case_j:
                assert all(x == -1 for x in orig_item["full_res"][resp_id])
                output_res = -1
            else:
                output_res = orig_item["full_res"][resp_id][case_j]
            output2res[case_j][str(case_o)] = output_res

    # Get self-consistency
    sc_res = True
    sc_outputs = []
    for case_j, output_cnt in enumerate(outputs_counter):
        if len(output_cnt) == 0:
            sc_res = False
            sc_outputs.append(None)
            continue

        sc_o = output_cnt.most_common(1)[0][0]
        sc_o_res = output2res[case_j][sc_o]
        if not sc_o_res:
            sc_res = False
        sc_outputs.append(sc_o)

    # Estimate the corresponding frequency rates
    tot_freq = 0
    for case_j, output_cnt in enumerate(outputs_counter):
        if len(output_cnt) == 0:
            continue

        max_freq = output_cnt.most_common(1)[0][1]
        tot_freq += max_freq

    tot_freq /= num_test_cases

    # Confirm if there is some program solution meets all self-consistency outputs
    sc_match_res = []
    for resp_id, (full_res, pg_outputs) in enumerate(zip(item["full_res"], item["outputs"])):
        resp_match_res = True
        for case_j, (case_r, case_o) in enumerate(zip(full_res, pg_outputs)):
            if case_j >= len(inputs):
                # resp_match_res = False
                break
            if case_r != 0:
                resp_match_res = False
                break
            # if not str(case_o):
            #     resp_match_res = False
            #     break

            if sc_outputs[case_j] is None:
                resp_match_res = False
                break

            if str(case_o) != sc_outputs[case_j]:
                resp_match_res = False
                if orig_item["full_res"][resp_id][case_j] and output2res[case_j][sc_outputs[case_j]]:
                    print(f"warning", str(case_o), sc_outputs[case_j], orig_item["test_cases"]["outputs"][case_j])
                break

        sc_match_res.append(resp_match_res)

    if sc_res and any(
            sc_match_res):  # If the self-consistency determined group is correct and there is one program match all predictions with self-consistency, then it is a correct solution
        prog_sc_res = True
    else:
        prog_sc_res = False

    return {
        "sc_res": sc_res,
        "sc_outputs": sc_outputs,
        "tot_freq": tot_freq,
        "prog_sc_res": prog_sc_res,
        "sc_match_res": sc_match_res,
        "res": orig_item["res"][0],
        "id": item["id"]
    }


def load_files(file_path):
    data = []
    if os.path.exists(file_path):
        if file_path.endswith(".json"):
            data.extend(json.load(open(file_path)))
        else:
            data.extend([json.loads(line) for line in open(file_path).readlines()])
    else:
        for file in glob(file_path):
            print(file)
            if file.endswith(".json"):
                data.extend(json.load(open(file)))
            else:
                data.extend([json.loads(line) for line in open(file).readlines()])

    return data


def merge_key(item, value):
    assert isinstance(item, list)
    if isinstance(value, list):
        item = item + value
    else:
        item.append(value)
    return item


def merge_seed_sampled_data(data):
    id2data = {}
    for item in data:
        if item["id"] not in id2data:
            id2data[item["id"]] = item
            continue

        tmp = id2data[item["id"]]
        if isinstance(tmp["response"], str):
            tmp["response"] = [tmp["response"]]
        if not isinstance(tmp["res"], list):
            tmp["res"] = [tmp["res"]]
        if not isinstance(tmp["pred"], list):
            tmp["pred"] = [tmp["pred"]]
        if not isinstance(tmp["full_res"], list):
            tmp["full_res"] = [tmp["full_res"]]
        if "outputs" in tmp and not isinstance(tmp["outputs"], list):
            tmp["outputs"] = [tmp["outputs"]]

        tmp["response"] = merge_key(tmp["response"], item["response"])
        tmp["res"] = merge_key(tmp["res"], item["res"])
        tmp["pred"] = merge_key(tmp["pred"], item["pred"])
        tmp["full_res"] = merge_key(tmp["full_res"], item["full_res"])
        if "outputs" in tmp:
            tmp["outputs"] = merge_key(tmp["outputs"], item["outputs"])
        assert isinstance(tmp["pred"], list), tmp["pred"]
        id2data[item["id"]] = tmp

    return list(id2data.values())


def main():
    parser = ArgumentParser()
    parser.add_argument("--completion_file", type=str)
    parser.add_argument("--exec_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    if not os.path.exists(f'apps_difficulty_{args.split}.json'):
        _dataset = load_dataset("codeparrot/apps", split=args.split).to_list()
        problem_id2difficulty = {item["problem_id"]: item["difficulty"] for item in _dataset}
        all_difficulties = collections.Counter(problem_id2difficulty.values())
        json.dump(problem_id2difficulty, open(f'apps_difficulty_{args.split}.json', "w"), ensure_ascii=False)
        json.dump(all_difficulties, open(f'apps_difficulty_{args.split}_all.json', "w"), ensure_ascii=False)
    else:
        problem_id2difficulty = json.load(open(f'apps_difficulty_{args.split}.json'))
        all_difficulties = json.load(open(f'apps_difficulty_{args.split}_all.json'))
        problem_id2difficulty = {int(k): v for k, v in problem_id2difficulty.items()}

    completions = load_files(args.completion_file)
    completions = merge_seed_sampled_data(completions)
    execs = load_files(args.exec_file)
    execs = merge_seed_sampled_data(execs)
    print(len(completions), len(execs))

    id2completion = {item["id"]: item for item in completions}
    id2exec = {item["id"]: item for item in execs}
    print(len(id2completion), len(id2exec))

    commons = set(id2completion.keys()) & set(id2exec.keys())
    print(f"Found {len(commons)} common items.")

    inputs = []
    for _id in commons:
        inputs.append((id2exec[_id], id2completion[_id]))

    pbar = tqdm(inputs)
    outputs = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        _annotate = worker
        for _input in pbar:
            future = executor.submit(_annotate, _input)
            futures.append(future)
            pbar.update()

        for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting results"):
            result = future.result()
            result["difficulty"] = problem_id2difficulty[result["id"]]
            outputs.append(result)

    json.dump(outputs, open(args.output_file, "w"))

    sc = 0
    prog_sc = 0
    first_res = 0
    num_prog = 0
    for item in outputs:
        if item["sc_res"]:
            sc += 1
        if item["prog_sc_res"]:
            prog_sc += 1
        if item["res"]:
            first_res += 1
        num_prog += len(item["sc_match_res"])

    print(f"Self-consistency: {sc}/{len(completions)} = {sc / len(completions)}")
    print(f"Program self-consistency: {prog_sc}/{len(completions)} = {prog_sc / len(completions)}")
    print(f"First res: {first_res}/{len(completions)} = {first_res / len(completions)}")
    print(f"Programs: {num_prog}/{len(completions)} = {num_prog / len(completions)}")


if __name__ == "__main__":
    main()
