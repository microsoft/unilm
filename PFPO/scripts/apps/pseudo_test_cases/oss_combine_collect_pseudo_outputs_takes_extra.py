import collections
import functools
import json
import os
import sys
from argparse import ArgumentParser
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from pympler import asizeof

from tqdm import tqdm

sys.set_int_max_str_digits(0)

"""
Copied from `scripts/apps/pseudo_test_cases/oss_combine_collect_pseudo_outputs.py`.

This script can accept external execution files for performing self-consistency over test cases.
"""


def worker(_input, min_success_test_num: int = 1, top_p: float = 0.0):
    item, sc_item = _input
    inputs = sc_item["input_output"]["inputs"]
    outputs_counter = [
        collections.Counter() for _ in inputs
    ]
    output_str2orig_pred = [
        {} for _ in inputs
    ]
    resp2outputs = [
        {} for _ in range(len(item["outputs"]))
    ]
    assert len(inputs) == len(item["input_output"]["inputs"]), (len(inputs), len(item["input_output"]["inputs"]))
    # assert len(sc_item["outputs"]) == item["outputs"], (len(sc_item["outputs"]), len(item["outputs"]))

    assert len(sc_item["full_res"]) == len(sc_item["outputs"]) == len(sc_item["pred"]), (len(sc_item["full_res"]), len(sc_item["outputs"]),
                                                                                         len(sc_item["pred"]))
    assert len(item["full_res"]) == len(item["outputs"]) == len(item["pred"]), (len(item["full_res"]), len(item["outputs"]), len(item["pred"]))

    sc_prog_num = len(sc_item["full_res"])

    # We use the external item to perform self-consistency over test cases.
    for resp_id, (full_res, pg_outputs) in enumerate(zip(sc_item["full_res"], sc_item["outputs"])):
        for case_j, (case_r, case_o) in enumerate(zip(full_res, pg_outputs)):
            if case_j >= len(inputs):
                break
            if case_r != 0:
                continue
            # assert case_o  # sometimes is could be `int` or `True` or `False`. We believe the `case_r` here.

            if not str(case_o):
                continue  # some outputs could be empty string

            if str(case_o) not in output_str2orig_pred[case_j]:
                output_str2orig_pred[case_j][str(case_o)] = case_o
            outputs_counter[case_j][str(case_o)] += 1

    # We then record the target item's outputs
    for resp_id, (full_res, pg_outputs) in enumerate(zip(item["full_res"], item["outputs"])):
        for case_j, (case_r, case_o) in enumerate(zip(full_res, pg_outputs)):
            if case_j >= len(inputs):
                break
            if case_r != 0:
                continue
            # assert case_o  # sometimes is could be `int` or `True` or `False`. We believe the `case_r` here.

            if not str(case_o):
                continue  # some outputs could be empty string

            resp2outputs[resp_id][case_j] = str(case_o)

    averaged_p = 0
    for case_j, output_cnt in enumerate(outputs_counter):
        if not output_cnt:
            continue
        sc_o_freq = output_cnt.most_common(1)[0][1]
        averaged_p += sc_o_freq / sc_prog_num

    averaged_p /= len(outputs_counter)

    if averaged_p < top_p:
        return {
            "id": item["id"],
            "inputs": [],
            "outputs": [],
        }

    new_inputs = []
    new_outputs = []
    new_output_meta = []
    sc_match_res = [[] for _ in range(len(item["pred"]))]
    for case_j, output_cnt in enumerate(outputs_counter):
        if not output_cnt:
            continue

        if sum(output_cnt.values()) < min_success_test_num:
            continue

        sc_o, sc_o_freq = output_cnt.most_common(1)[0]
        sc_o_real = output_str2orig_pred[case_j][sc_o]

        new_inputs.append(inputs[case_j])
        new_outputs.append(sc_o_real)

        new_output_meta.append({
            "output_freq": output_cnt,
            "output_str2orig_pred": output_str2orig_pred[case_j],
        })

        for pg_i in range(len(item["pred"])):
            if case_j not in resp2outputs[pg_i]:
                sc_match_res[pg_i].append(-2)  # compilation error
                continue
            if resp2outputs[pg_i][case_j] == sc_o:
                sc_match_res[pg_i].append(1)
            else:
                sc_match_res[pg_i].append(0)

    return {
        "id": item["id"],
        "inputs": new_inputs,
        "outputs": new_outputs,
        "output_meta": new_output_meta,
        "sc_match_res": sc_match_res,
    }


def load_files(file_path):
    data = []
    if os.path.exists(file_path):
        print(f"Loading pseudo test cases from {file_path}")
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
    large_mem = 0
    for item in data:
        if isinstance(item["response"], str):
            item["response"] = [item["response"]]
            assert isinstance(item["pred"], str) or item["pred"] is None
            item["pred"] = [item["pred"]]

        if "outputs" in item:
            size_in_bytes = asizeof.asizeof(item["outputs"])
            if size_in_bytes / (1024 ** 2) > 10:  # 10MB
                if "res" in item:
                    item.pop("res")
                if "full_res" in item:
                    item.pop("full_res")
                if "outputs" in item:
                    item.pop("outputs")
                if "errors" in item:
                    item.pop("errors")
                large_mem += 1

        if "res" not in item:  # Sometimes all solutions do not entail the programs. Please turn back to `solution_run_outputs_local.py`.
            results = []
            full_results = []
            all_outputs = []

            for _ in item["response"]:
                results.append(False)
                full_results.append([-2] * 21)
                all_outputs.append([None] * 21)

            item["res"] = results
            item["full_res"] = full_results
            item["outputs"] = all_outputs

        if item["id"] not in id2data:
            id2data[item["id"]] = item
            continue

        tmp = id2data[item["id"]]
        # if isinstance(tmp["res"], list):
        #     tmp["res"] = [tmp["res"]]
        # if not isinstance(tmp["pred"], list):
        #     tmp["pred"] = [tmp["pred"]]
        # if not isinstance(tmp["full_res"], list):
        #     tmp["full_res"] = [tmp["full_res"]]
        # if "outputs" in tmp and not isinstance(tmp["outputs"], list):
        #     tmp["outputs"] = [tmp["outputs"]]

        tmp["response"] = merge_key(tmp["response"], item["response"])
        tmp["res"] = merge_key(tmp["res"], item["res"])
        tmp["pred"] = merge_key(tmp["pred"], item["pred"])
        tmp["full_res"] = merge_key(tmp["full_res"], item["full_res"])
        tmp["outputs"] = merge_key(tmp["outputs"], item["outputs"])
        assert isinstance(tmp["pred"], list), tmp["pred"]
        id2data[item["id"]] = tmp

    print(f"Too large outputs: {large_mem}")
    return list(id2data.values())


def main():
    parser = ArgumentParser()
    parser.add_argument("--pseudo_test_case_file", type=str)
    parser.add_argument("--completion_file", type=str)
    parser.add_argument("--min_success_test_num", type=int, default=2)
    parser.add_argument("--pass_case_margin", type=float, default=1)
    parser.add_argument("--pass_case_lower_bound", type=float, default=0.5)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--top_p", type=float, default=0.0)
    args = parser.parse_args()

    external_ps_test_cases = load_files(args.pseudo_test_case_file)
    external_ps_test_cases = merge_seed_sampled_data(external_ps_test_cases)
    id2external_ps_test_cases = {item["id"]: item for item in external_ps_test_cases}

    data = load_files(args.completion_file)
    data = merge_seed_sampled_data(data)
    id2item = {item["id"]: item for item in data}

    before_test_num = 0
    missing_predictions = 0
    _mp_inputs = []
    _mp_outputs = []
    for item in tqdm(data):
        if "outputs" not in item:
            print(item["pred"])
            missing_predictions += 1
            continue

        before_test_num += len(item["input_output"]["inputs"])
        _mp_inputs.append((item, id2external_ps_test_cases[item["id"]]))

    pbar = tqdm(_mp_inputs)
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        _annotate = functools.partial(worker, min_success_test_num=args.min_success_test_num, top_p=args.top_p)
        for _input in pbar:
            future = executor.submit(_annotate, _input)
            futures.append(future)
            pbar.update()

        for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting results"):
            _mp_outputs.append(future.result())

    outputs = []
    cnt = 0
    avg_test_case_num = 0
    pass_cnt = collections.Counter()
    for result in _mp_outputs:
        if not result["inputs"]:
            continue

        item = id2item[result["id"]]

        item["input_output"]["inputs"] = result["inputs"]
        item["input_output"]["outputs"] = result["outputs"]
        item["input_output"]["output_meta"] = result["output_meta"]
        item["sc_full_res"] = result["sc_match_res"]
        avg_test_case_num += len(result["inputs"])

        pred_pass_cnt = []
        for pg_i, pg_res in enumerate(item["sc_full_res"]):
            pred_pass_cnt.append(sum([1 for r in pg_res if r == 1]))
            pass_cnt[pred_pass_cnt[-1]] += 1

        pos = []
        neg = []
        pos_code = []
        neg_code = []
        num_test_cases = len(item["input_output"]["inputs"])
        assert num_test_cases == len(item["sc_full_res"][0])
        assert len(pred_pass_cnt) == len(item["response"]) == len(item["pred"])
        for i in range(len(pred_pass_cnt)):
            resp_i = item["response"][i]
            prog_i = item["pred"][i]
            pass_cnt_i = pred_pass_cnt[i]
            if pass_cnt_i / num_test_cases < args.pass_case_lower_bound:
                continue
            for j in range(len(pred_pass_cnt)):
                if i == j:
                    continue
                resp_j = item["response"][j]
                prog_j = item["pred"][j]
                pass_cnt_j = pred_pass_cnt[j]
                if pass_cnt_i - pass_cnt_j >= args.pass_case_margin:
                    pos.append(resp_i)
                    pos_code.append(prog_i)
                    neg.append(resp_j)
                    neg_code.append(prog_j)

        item["pos"] = pos
        item["pos_code"] = pos_code
        item["neg"] = neg
        item["neg_code"] = neg_code
        cnt += len(pos)

        outputs.append(item)

    print(len(outputs))
    print(cnt)
    print(missing_predictions)
    print(before_test_num / len(data) if data else 0)
    print(avg_test_case_num / len(outputs) if outputs else 0)
    print(pass_cnt)

    json.dump(outputs, open(args.output_file, "w"), indent=2)


if __name__ == "__main__":
    main()
