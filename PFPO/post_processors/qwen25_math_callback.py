from post_processors.openai_api_callback import OpenAICallBack, majority_voting_predict
import collections
import json
import os
import re
from typing import Dict, Any, List, Tuple, Union
import numpy as np
import vllm
from data.qwen25math.parser import extract_answer, strip_string, STRIP_EXCEPTIONS
from data.qwen25math.grader import math_equal_process, math_equal
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from omegaconf import ListConfig

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


def _annotate(param):
    return param[0], math_equal(param[-2], param[-1])


class Qwen25MathCallBack(OpenAICallBack):
    def __init__(self, *args, num_workers: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_workers = num_workers

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], **kwargs):
        text = meta_data["text"]
        if self.label_field in meta_data:
            label = meta_data[self.label_field]
        else:
            label = -1
        index = meta_data[self.index_field]

        response = batch_model_outputs["response"]
        if isinstance(response, vllm.RequestOutput):
            if response.finished:
                response = [o.text for o in response.outputs]
                if len(response) == 1:
                    response = response[0]
            else:
                response = ""

        if isinstance(response, str):
            pred_clean = extract_answer(response, data_name="math")
            pred_clean = strip_string(pred_clean, skip_unit="math" in STRIP_EXCEPTIONS)
            if pred_clean is None:
                pred_clean = ""
            sc_pred = pred_clean
        elif isinstance(response, list):
            pred_clean = []
            for resp in response:
                tmp_pred_clean = extract_answer(resp, data_name="math")
                tmp_pred_clean = strip_string(tmp_pred_clean, skip_unit="math" in STRIP_EXCEPTIONS)
                if tmp_pred_clean is None:
                    tmp_pred_clean = ""
                pred_clean.append(tmp_pred_clean)
            sc_pred = majority_voting_predict(pred_clean)
        else:
            raise ValueError(f"Unknown type of response: {type(response)}")

        out_item = {
            "text": text,
            "label": label,
            "response": response,
            "pred": pred_clean,
            "id": index,
            "sc_pred": sc_pred,
        }
        if self.saved_keys is not None:
            for key in self.saved_keys:
                out_item[key] = meta_data[key]
        self.predictions.append(out_item)
        self.fw.write(json.dumps(self.predictions[-1]) + "\n")
        self.fw.flush()

    def get_results(self):
        save_dir = os.path.dirname(self.output_file)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        self.fw.close()

        _mp_inputs = []
        for i, item in enumerate(self.predictions):
            if not isinstance(item["pred"], list):
                preds = [item["pred"]]
            else:
                preds = item["pred"]
            for j, pred in enumerate(preds):
                _mp_inputs.append(((i, j), pred, str(item["label"])))
        pbar = tqdm(_mp_inputs, total=len(_mp_inputs), desc="Submitting eval task", dynamic_ncols=True)

        outputs = collections.defaultdict(dict)
        timeout_cnt = 0
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for _input in pbar:
                future = executor.submit(_annotate, _input)
                futures.append(future)
                pbar.update()

            for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting results"):
                try:
                    idx, result = future.result()
                    outputs[idx[0]][idx[1]] = result
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    outputs[idx[0]][idx[1]] = False
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()

        for i, item in enumerate(self.predictions):
            if not isinstance(item["pred"], list):
                preds = [item["pred"]]
            else:
                preds = item["pred"]
            all_res = outputs[i]
            assert len(all_res) == len(preds)
            pred2res = {pred: all_res[j] for j, pred in enumerate(preds)}
            sc_res = pred2res[item["sc_pred"]]

            item["res"] = [pred2res[pred] for pred in preds]
            item["sc_res"] = sc_res

            if not isinstance(item["pred"], list):
                assert len(item["res"]) == 1
                item["res"] = item["res"][0]

        cnt = 0
        pass_at_k = 0
        sc = 0
        acc_data_topic = collections.Counter()
        cnt_data_topic = collections.Counter()
        for item in self.predictions:
            if not isinstance(item["res"], list):
                res = [item["res"]]
            else:
                res = item["res"]
            if res[0]:
                cnt += 1
            if "data_topic" in item:
                if "." in item["data_topic"]:
                    item["data_topic"] = item["data_topic"].split(".")[0]
                acc_data_topic[item["data_topic"]] += int(res[0])
                cnt_data_topic[item["data_topic"]] += 1
            if any(res):
                pass_at_k += 1
            if item["sc_res"]:
                sc += 1

        assert pass_at_k <= len(self.predictions)
        json.dump(self.predictions, open(self.output_file, "w"), indent=2)

        if len(self.predictions) == 0:
            metrics = {"acc": 0, "pass@k": 0, "maj@k": 0, "correct": 0, "total": 0}
        else:
            metrics = {"acc": cnt / len(self.predictions), "pass@k": pass_at_k / len(self.predictions), "maj@k": sc / len(self.predictions),
                       "correct": cnt, "total": len(self.predictions)}
            if len(acc_data_topic) > 0:
                for key in acc_data_topic:
                    metrics[f"acc_{key}"] = acc_data_topic[key] / cnt_data_topic[key]
        json.dump(metrics, open(self.output_file.replace(".json", ".metrics.json"), "w"), indent=2)
        return metrics, []
