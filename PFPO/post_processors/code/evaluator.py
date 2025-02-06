import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from tqdm import tqdm

from eval.codex_humaneval.execution import check_correctness as humaneval_check_correctness
from scripts.apps.utils_execute import check_correctness as apps_check_correctness
from eval.mbpp_eval.execute import check_correctness as mbpp_check_correctness


def return_apps_evaluator(timeout: int = 10, debug: bool = False):
    return partial(apps_check_correctness, timeout=timeout, debug=debug)


class HumanEvaluator:
    def __init__(self, ):
        pass

    def __call__(self, predictions, num_workers: int = 16):
        success = 0
        success_at_k = 0

        evaluator = partial(humaneval_check_correctness, timeout=10)
        # Multiprocessing
        _mp_inputs = []
        for i, item in enumerate(predictions):
            if item["test_cases"]:
                if isinstance(item["pred"], list):
                    preds = item["pred"]
                else:
                    preds = [item["pred"]]

                # item["res"] = [False for _ in range(len(preds))]

                for j, pred in enumerate(preds):
                    # _mp_inputs.append(((i, j), item["test_cases"], pred))
                    if pred:
                        _mp_inputs.append({
                            "problem": {
                                "prompt": item["prompt"],
                                "test": item["test_cases"],
                                "entry_point": item["entry_point"],
                                "task_id": item["id"],
                            },
                            "completion": pred,
                            "completion_id": (i, j)
                        })

        pbar = tqdm(_mp_inputs, total=len(_mp_inputs), desc="Evaluating", dynamic_ncols=True)

        if len(_mp_inputs) > 0:
            outputs = collections.defaultdict(dict)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for _input in pbar:
                    future = executor.submit(evaluator, **_input)
                    futures.append(future)
                    pbar.update()

                for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting results"):
                    res = future.result()
                    outputs[res["completion_id"][0]][res["completion_id"][1]] = res

            for i, item in enumerate(predictions):
                if item["test_cases"]:
                    if isinstance(item["pred"], list):
                        preds = item["pred"]
                    else:
                        preds = [item["pred"]]

                    res = []
                    for j, pred in enumerate(preds):
                        if pred:
                            program_res = outputs[i][j]["passed"]
                            res.append(program_res)
                        else:
                            res.append(False)

                    if any(res):
                        success_at_k += 1
                    if res[0]:
                        success += 1

                    if len(preds) == 1:
                        item["res"] = res[0]
                    else:
                        item["res"] = res
                else:
                    item["res"] = []

        if len(predictions) == 0:
            metrics = {"acc": 0, "pass@k": 0, "correct": 0, "total": 0}
        else:
            metrics = {"acc": success / len(predictions), "pass@k": success_at_k / len(predictions), "correct": success,
                       "total": len(predictions)}
        return predictions, metrics


class MBPPEvaluator:
    def __init__(self, ):
        pass

    def __call__(self, predictions, num_workers: int = 16):
        success = 0
        success_at_k = 0

        evaluator = partial(mbpp_check_correctness, timeout=10)
        # Multiprocessing
        _mp_inputs = []
        for i, item in enumerate(predictions):
            if item["test_cases"]:
                if isinstance(item["pred"], list):
                    preds = item["pred"]
                else:
                    preds = [item["pred"]]

                for j, pred in enumerate(preds):
                    if pred:
                        _mp_inputs.append({
                            "check_program": pred + "\n" + item["test_cases"],
                            "task_id": item["id"],
                            "completion_id": (i, j)
                        })

        pbar = tqdm(_mp_inputs, total=len(_mp_inputs), desc="Evaluating", dynamic_ncols=True)

        if len(_mp_inputs) > 0:
            outputs = collections.defaultdict(dict)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for _input in pbar:
                    future = executor.submit(evaluator, **_input)
                    futures.append(future)
                    pbar.update()

                for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting results"):
                    res = future.result()
                    outputs[res["completion_id"][0]][res["completion_id"][1]] = res

            for i, item in enumerate(predictions):
                if item["test_cases"]:
                    if isinstance(item["pred"], list):
                        preds = item["pred"]
                    else:
                        preds = [item["pred"]]

                    res = []
                    for j, pred in enumerate(preds):
                        if pred:
                            program_res = outputs[i][j]["passed"]
                            res.append(program_res)
                        else:
                            res.append(False)

                    if any(res):
                        success_at_k += 1
                    if res[0]:
                        success += 1

                    if len(preds) == 1:
                        item["res"] = res[0]
                    else:
                        item["res"] = res
                else:
                    item["res"] = []

        if len(predictions) == 0:
            metrics = {"acc": 0, "pass@k": 0, "correct": 0, "total": 0}
        else:
            metrics = {"acc": success / len(predictions), "pass@k": success_at_k / len(predictions), "correct": success,
                       "total": len(predictions)}
        return predictions, metrics
