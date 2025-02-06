import collections
import io
import json
import os
import re
from typing import Dict, Any, List, Tuple, Union
import numpy as np
import vllm
from data import math_util
from data.deepseek_math_utils import eval_script, answer_extraction
from data.mathscale.util import mathscale_is_equiv_proxy, is_correct as mathscale_is_correct
from omegaconf import ListConfig

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class PlaceholderClean:
    def __call__(self, pred: str):
        return "A"


class MCQAAnswerClean:
    def __init__(self, prompt: str = "zero-shot"):
        self.prompt = prompt

    def __call__(self, pred: str):
        # print("pred_before: ", pred)
        preds = re.findall(r"A|B|C|D|E", pred)
        if len(preds) == 0:
            return ""

        if self.prompt == "zero-shot":
            return preds[0]
        if self.prompt == "few-shot":
            return preds[-1]
        return preds[0]


class SeparatorClean:
    def __init__(self, separator: str = "Finish", separate_idx: int = 1, regrex: str = "A|B|C|D"):
        self.separator = separator
        self.separate_idx = separate_idx
        self.regrex = re.compile(regrex)

    def __call__(self, pred: str):
        if self.separator and self.separator in pred:
            preds = pred.split(self.separator)
            if len(preds) == 0:
                return ""

            if len(preds) <= self.separate_idx:
                return ""

            pred = preds[self.separate_idx]

        preds = re.findall(self.regrex, pred)
        if len(preds) == 0 or len(preds) > 1:
            return ""
        return preds[0]


class ReActSeparatorClean:  # FIXED@2024-01-03: Add hard constraint.
    def __init__(self, separator: str = "Context:", separate_idx: int = 0, regrex: str = "A|B|C|D"):
        self.separator = separator  # Use for remove generated dummy examples
        self.separate_idx = separate_idx
        self.regrex = re.compile(regrex)

    def __call__(self, pred: str):
        if self.separator in pred:
            groups = pred.split(self.separator)
            pred = groups[self.separate_idx]

        if "Finish[" in pred:
            pred = pred.split("Finish[")[1]
            pred = pred.split("]")[0]
            preds = re.findall(self.regrex, pred)
            if len(preds) == 0:
                return ""
            elif len(preds) == 1:
                return preds[0]
            else:
                return ""  # FIXED@2023-12-27: To avoid the case where the large language models tends to generate multiple predictions to hack the answer.
        return ""


class BinaryAnswerClean:
    def __init__(self, prompt: str = "zero-shot"):
        self.prompt = prompt

    def __call__(self, pred: str):
        preds = re.findall(r"Yes|No", pred)
        if len(preds) == 0:
            return ""

        if self.prompt == "zero-shot":
            return preds[0]
        if self.prompt == "few-shot":
            return preds[-1]
        return preds[0]


class TagCleaner:
    def __call__(self, pred: str):
        # Regular expression pattern to match the content between <answer> and </answer>
        pattern = r'<answer>(.*?)</answer>'
        # Use re.DOTALL to allow matching newlines within the tags
        match = re.search(pattern, pred, re.DOTALL)
        if match:
            return match.group(1).strip()  # Strip removes extra spaces or newlines
        return pred


class OpenAICallBack:
    def __init__(self, output_file: str, answer_clean: Union[MCQAAnswerClean, str], resume: bool = False, index_field: str = "index",
                 label_field: str = "label", saved_keys: List[str] = None):
        self.predictions = []
        self.output_file = output_file
        self.answer_clean = answer_clean
        self.index_field = index_field
        self.label_field = label_field
        self.saved_keys = saved_keys
        if isinstance(self.saved_keys, ListConfig):
            self.saved_keys = list(self.saved_keys)

        logging_file = output_file.replace(".json", ".jsonl")
        save_dir = os.path.dirname(logging_file)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        if os.path.exists(logging_file):
            if resume:
                with open(logging_file, "r") as f:
                    for line in f.readlines():
                        # self.predictions.append(json.loads(line))
                        item = json.loads(line)
                        if isinstance(item["response"], str):
                            if item["response"].strip() == "":
                                continue
                        elif isinstance(item["response"], list):
                            if any([tmp.strip() == "" for tmp in item["response"]]):
                                continue
                        self.predictions.append(item)
                logger.info(f"Load {len(self.predictions)} from {logging_file}")

        self.logging_file = logging_file

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], fw: io = None, **kwargs):
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
            pred_clean = self.answer_clean(response)
        elif isinstance(response, list):
            pred_clean = [self.answer_clean(item) for item in response]
        else:
            raise ValueError(f"Unknown type of response: {type(response)}")

        out_item = {
            "text": text,
            "label": label,
            "response": response,
            "pred": pred_clean,
            "id": index,
        }
        if self.saved_keys is not None:
            for key in self.saved_keys:
                if key in meta_data:
                    out_item[key] = meta_data[key]
        self.predictions.append(out_item)
        if fw is not None:
            fw.write(json.dumps(self.predictions[-1]) + "\n")
        else:
            with open(self.logging_file, "a") as f:
                f.write(json.dumps(self.predictions[-1]) + "\n")

    def batch_call(self, meta_data: List[Dict[str, Any]], batch_model_outputs: List[Dict[str, Any]], **kwargs):
        with open(self.logging_file, "a") as f:
            for m, b in zip(meta_data, batch_model_outputs):
                self(m, b, fw=f, **kwargs)

    @staticmethod
    def eval_single_item(pred, label):
        if not pred.strip():
            return False
        if len(pred.strip()) > 1:
            return False
        if isinstance(label, str):
            if label.strip() == pred.strip():
                return True
        if isinstance(label, list) and isinstance(label[0], str):
            if label[0].strip() == pred.strip():
                return True
        if label == ord(pred.strip()) - ord("A"):
            return True
        return False

    def get_results(self):
        save_dir = os.path.dirname(self.output_file)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        json.dump(self.predictions, open(self.output_file, "w"), indent=2)

        cnt = 0
        outputs = []
        pass_at_k = 0
        for item in self.predictions:
            if isinstance(item["pred"], list):
                preds = item["pred"]
            else:
                preds = [item["pred"]]

            pred = collections.Counter(preds).most_common(1)[0][0]

            mul_pass = 0
            for tmp in preds:
                if self.eval_single_item(tmp, item["label"]):
                    mul_pass = 1
                    break
            pass_at_k += mul_pass

            if not pred.strip():
                outputs.append((item["id"], 0))
                continue
            if len(pred.strip()) > 1:
                outputs.append((item["id"], 0))
                continue
            if isinstance(item["label"], str):
                if item["label"].strip() == pred.strip():
                    cnt += 1
            elif isinstance(item["label"], list) and isinstance(item["label"][0], str):
                if item["label"][0].strip() == pred.strip():
                    cnt += 1
            else:
                if item["label"] == ord(pred.strip()) - ord("A"):
                    cnt += 1
            outputs.append((item["id"], ord(pred.strip()) - ord("A")))
        assert len(outputs) == len(self.predictions)

        # Remove duplicated ids to satisfy the submission requirements of ReClor.
        outputs = sorted(outputs, key=lambda x: x[0])
        id_set = set()
        new_outputs = []
        for item in outputs:
            if item[0] not in id_set:
                new_outputs.append(item[1])
                id_set.add(item[0])
        outputs = new_outputs

        np_output_file = self.output_file.replace(".json", ".npy")
        np.save(np_output_file, np.array(outputs))

        if len(self.predictions) == 0:
            metrics = {"acc": 0, "pass@k": 0, "correct": 0, "total": 0}
        else:
            metrics = {"acc": cnt / len(self.predictions), "pass@k": pass_at_k / len(self.predictions), "correct": cnt, "total": len(self.predictions)}
        json.dump(metrics, open(self.output_file.replace(".json", ".metrics.json"), "w"), indent=2)
        return metrics, []


class SaveOnlyCallBack(OpenAICallBack):
    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], fw: io = None, **kwargs):
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

        out_item = {
            "text": text,
            "label": label,
            "response": response,
            "id": index,
        }
        if self.saved_keys is not None:
            for key in self.saved_keys:
                out_item[key] = meta_data[key]
        self.predictions.append(out_item)
        # self.fw.write(json.dumps(self.predictions[-1]) + "\n")
        # self.fw.flush()
        if fw is not None:
            fw.write(json.dumps(self.predictions[-1]) + "\n")
        else:
            with open(self.logging_file, "a") as f:
                f.write(json.dumps(self.predictions[-1]) + "\n")

    def get_results(self):
        save_dir = os.path.dirname(self.output_file)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        json.dump(self.predictions, open(self.output_file, "w"), indent=2)
        # self.fw.close()
        return {}, []


def majority_voting_predict(preds):
    if isinstance(preds, str):
        return preds

    preds = [pred for pred in preds if pred]
    if len(preds) == 0:
        return ""

    assert isinstance(preds, list)
    if isinstance(preds[0], list):
        tmp = []
        for pred in preds:
            tmp.append(str(sorted(pred)))
        pred = collections.Counter(tmp).most_common(1)[0][0]
        pred = eval(pred)
    elif isinstance(preds[0], str):
        pred = collections.Counter(preds).most_common(1)[0][0]
    else:
        # raise ValueError(f"Unknown type {type(preds[0])}")
        logger.warning(f"Unknown type {type(preds[0])}")
        pred = ""
    return pred


class OpenAIMATHCallBack(OpenAICallBack):
    eval_fns = {
        "meta_math": math_util.is_equiv,
    }

    def __init__(self, *args, eval_fn: str = "meta_math", **kwargs):
        super().__init__(*args, **kwargs, )
        self.eval_fn = self.eval_fns[eval_fn]

    def get_results(self):
        save_dir = os.path.dirname(self.output_file)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        self.fw.close()

        cnt = 0
        pass_at_k = 0
        sc = 0
        outputs = []
        for item in self.predictions:
            if isinstance(item["pred"], list):
                preds = item["pred"]
            else:
                preds = [item["pred"]]

            # pred = collections.Counter(preds).most_common(1)[0][0]

            # res = math_util.is_equiv(pred, item["label"])
            # res = [math_util.is_equiv(p, item["label"]) for p in preds]
            res = [self.eval_fn(p, item["label"]) for p in preds]
            if isinstance(res[0], tuple):
                res = [r[0] for r in res]
            if res[0]:
                cnt += 1
            if any(res):
                pass_at_k += 1
            if isinstance(item["pred"], str):
                res = res[0]
            item["res"] = res

            sc_pred = majority_voting_predict(preds)
            sc_res = self.eval_fn(sc_pred, item["label"])
            item["sc_res"] = sc_res
            item["sc_pred"] = sc_pred
            if sc_res:
                sc += 1

            outputs.append((item["id"], res))

        assert len(outputs) == len(self.predictions)

        # Remove duplicated ids to satisfy the submission requirements of ReClor.
        # outputs = sorted(outputs, key=lambda x: x[0])
        # id_set = set()
        # new_outputs = []
        # for item in outputs:
        #     if item[0] not in id_set:
        #         new_outputs.append(item[1])
        #         id_set.add(item[0])
        # outputs = new_outputs

        json.dump(self.predictions, open(self.output_file, "w"), indent=2)

        if len(self.predictions) == 0:
            metrics = {"acc": 0, "pass@k": 0, "maj@k": 0, "correct": 0, "total": 0}
        else:
            metrics = {"acc": cnt / len(self.predictions), "pass@k": pass_at_k / len(self.predictions), "maj@k": sc / len(self.predictions),
                       "correct": cnt, "total": len(self.predictions)}
        json.dump(metrics, open(self.output_file.replace(".json", ".metrics.json"), "w"), indent=2)
        return metrics, []


class DeepSeekMathCallBack(OpenAICallBack):
    eval_fns = {
        "gsm8k": eval_script.eval_last_single_answer,
        "math": eval_script.eval_math,
    }

    extract_fns = {
        "gsm8k": answer_extraction.extract_last_single_answer,
        "math": answer_extraction.extract_math_answer,
    }

    def __init__(self, *args, eval_fn: str = "gsm8k", **kwargs):
        super().__init__(*args, **kwargs, )
        self.eval_fn = self.eval_fns[eval_fn]
        if self.answer_clean in self.extract_fns:
            self.extract_fn = self.extract_fns[self.answer_clean]
        else:
            self.extract_fn = self.answer_clean

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], fw: io = None, **kwargs):
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
            pred_clean = self.extract_fn(text, response, "cot")
        elif isinstance(response, list):
            pred_clean = [self.extract_fn(text, item, "cot") for item in response]
        else:
            raise ValueError(f"Unknown type of response: {type(response)}")

        out_item = {
            "text": text,
            "label": label,
            "response": response,
            "pred": pred_clean,
            "id": index,
        }
        if self.saved_keys is not None:
            for key in self.saved_keys:
                out_item[key] = meta_data[key]
        self.predictions.append(out_item)
        if fw is not None:
            fw.write(json.dumps(self.predictions[-1]) + "\n")
        else:
            with open(self.logging_file, "a") as f:
                f.write(json.dumps(self.predictions[-1]) + "\n")

    def get_results(self):
        save_dir = os.path.dirname(self.output_file)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        self.fw.close()

        cnt = 0
        pass_at_k = 0
        sc = 0
        outputs = []
        num_errors = 0
        for item in self.predictions:
            if isinstance(item["response"], list):
                preds = item["pred"]
            else:
                preds = [item["pred"]]

            mul_pass = 0
            if len(preds) > 0:
                # if isinstance(preds[0], list):
                #     # pred = preds[0]  # TODO: How to add self-consistency for MATH dataset given the answer could be a list?
                #     tmp = [str(x) for x in preds]
                #     pred = collections.Counter(tmp).most_common(1)[0][0]
                #     pred = eval(pred)
                # else:
                #     pred = collections.Counter(preds).most_common(1)[0][0]
                #
                # if pred is None:
                #     pred = ""
                res = []

                # res = math_util.is_equiv(pred, item["label"])
                # https://github.com/deepseek-ai/DeepSeek-Math/blob/main/evaluation/infer/run_pal_eval.py#L168
                # https://github.com/deepseek-ai/DeepSeek-Math/blob/main/evaluation/infer/run_cot_eval.py#L120
                # res = self.eval_fn({"prediction": pred, "answer": item["label"]})  # For CoT eval, use `prediction`, for Pal eval, use `program_output`.
                for pred in preds:
                    res.append(self.eval_fn({"prediction": pred, "answer": item["label"]}))

                # mul_pass = 0
                # for pred in preds:
                #     if self.eval_fn({"prediction": pred, "answer": item["label"]}):
                #         mul_pass = 1
                #         break
                if any(res):
                    mul_pass = 1

                sc_pred = majority_voting_predict(preds)
                item["sc_pred"] = sc_pred
                try:
                    sc_res = self.eval_fn({"prediction": sc_pred, "answer": item["label"]})
                    item["sc_res"] = sc_res
                except Exception as e:
                    logger.warning(f"Error in {item['id']} during evaluation: {e}")
                    sc_res = False
                    num_errors += 1
                if sc_res:
                    sc += 1
            else:
                res = []
                item["sc_pred"] = ""
                item["sc_res"] = False

            item["pass_at_k"] = mul_pass

            if len(res) > 0 and res[0]:
                cnt += 1
            if mul_pass:
                pass_at_k += 1
            outputs.append((item["id"], res))

            if len(preds) == 1:
                res = res[0]
            item["res"] = res

        assert len(outputs) == len(self.predictions)
        assert pass_at_k <= len(self.predictions)
        json.dump(self.predictions, open(self.output_file, "w"), indent=2)

        logger.info(f"Number of errors: {num_errors}")

        # Remove duplicated ids to satisfy the submission requirements of ReClor.
        # outputs = sorted(outputs, key=lambda x: x[0])
        # id_set = set()
        # new_outputs = []
        # for item in outputs:
        #     if item[0] not in id_set:
        #         new_outputs.append(item[1])
        #         id_set.add(item[0])
        # outputs = new_outputs

        if len(self.predictions) == 0:
            metrics = {"acc": 0, "pass@k": 0, "maj@k": 0, "correct": 0, "total": 0}
        else:
            metrics = {"acc": cnt / len(self.predictions), "pass@k": pass_at_k / len(self.predictions), "maj@k": sc / len(self.predictions),
                       "correct": cnt, "total": len(self.predictions)}
        json.dump(metrics, open(self.output_file.replace(".json", ".metrics.json"), "w"), indent=2)
        return metrics, []


class MathScaleCallBack(OpenAICallBack):

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], fw: io = None, **kwargs):
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
            if self.answer_clean is not None:
                resp_clean = self.answer_clean(response)
                if resp_clean is None or resp_clean is False:
                    resp_clean = ""
            else:
                resp_clean = response
            res, pred_clean, _ = mathscale_is_correct(resp_clean, label)
            if pred_clean is None:
                pred_clean = ""
            sc_pred = pred_clean
            sc_res = res
        elif isinstance(response, list):
            res = []
            pred_clean = []
            for item in response:
                if self.answer_clean is not None:
                    tmp_resp_clean = self.answer_clean(item)
                else:
                    tmp_resp_clean = item
                tmp_res, tmp_pred_clean, _ = mathscale_is_correct(tmp_resp_clean, label)
                if tmp_pred_clean is None:
                    tmp_pred_clean = ""
                res.append(tmp_res)
                pred_clean.append(tmp_pred_clean)
            pred2res = {pred: r for pred, r in zip(pred_clean, res)}
            sc_pred = majority_voting_predict(pred_clean)
            sc_res = pred2res[sc_pred]
        else:
            raise ValueError(f"Unknown type of response: {type(response)}")

        out_item = {
            "text": text,
            "label": label,
            "response": response,
            "pred": pred_clean,
            "id": index,
            "res": res,
            "sc_pred": sc_pred,
            "sc_res": sc_res,
        }
        if self.saved_keys is not None:
            for key in self.saved_keys:
                if key in meta_data:
                    out_item[key] = meta_data[key]
        self.predictions.append(out_item)
        if fw is not None:
            fw.write(json.dumps(self.predictions[-1]) + "\n")
        else:
            with open(self.logging_file, "a") as f:
                f.write(json.dumps(self.predictions[-1]) + "\n")

    def get_results(self):
        save_dir = os.path.dirname(self.output_file)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

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


def fix_trailing_comma(json_string):
    # Use regex to find and remove trailing commas before a closing brace or bracket
    fixed_string = re.sub(r',\s*(\}|\])', r'\1', json_string)
    return fixed_string


def extract_json_content_rep(input_str):
    """
    Extracts JSON content from a string, removing surrounding ``` symbols
    and an optional 'json' tag.

    Args:
        input_str (str): The input string containing JSON content.

    Returns:
        str: Cleaned JSON string.
    """
    # Remove leading and trailing ``` symbols
    cleaned_str = re.sub(r"^```(?:json)?|```$", "", input_str.strip())
    return fix_trailing_comma(cleaned_str.strip())


def extract_json_content(input_str):
    """
    Extracts JSON content from a string by isolating the portion enclosed
    between the ``` markers, optionally preceded by the 'json' tag.

    Args:
        input_str (str): The input string containing JSON content.

    Returns:
        str: Extracted JSON string or an empty string if no JSON is found.
    """
    # Use regex to find content between triple backticks
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", input_str, re.DOTALL)
    if match:
        return fix_trailing_comma(match.group(1).strip())
    return ""  # Return an empty string if no match is found


class JsonObjEvalCallBack(OpenAICallBack):
    @staticmethod
    def json_parse_and_eval(response: str, label: dict):
        if not response:
            return False, {}
        try:
            json_str = extract_json_content(response)
            if json_str == "":
                json_str = extract_json_content_rep(response)
            json_obj = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(e)
            return False, {}

        for k, v in label.items():
            if k not in json_obj:
                return False, json_obj
            if json_obj[k] != v:
                return False, json_obj
        return True, json_obj

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], fw: io = None, **kwargs):
        text = meta_data["text"]
        if self.label_field and self.label_field in meta_data:
            label = meta_data[self.label_field]
            if not isinstance(label, dict):
                try:
                    label = json.loads(label)
                except Exception as e:
                    logger.warning(f"Error in label when passing string: {e}")
                    logger.warning(label)
                    label = {}
        else:
            label = {}
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
            res, pred_clean = self.json_parse_and_eval(response, label)
        elif isinstance(response, list):
            res, pred_clean = [], []
            for resp in response:
                tmp_res, tmp_pred_clean = self.json_parse_and_eval(resp, label)
                res.append(tmp_res)
                pred_clean.append(tmp_pred_clean)
        else:
            raise ValueError(f"Unknown type of response: {type(response)}")

        out_item = {
            "text": text,
            "label": label,
            "response": response,
            "id": index,
            "res": res,
            "pred": pred_clean,
        }
        if self.saved_keys is not None:
            for key in self.saved_keys:
                if key in meta_data:
                    out_item[key] = meta_data[key]
        self.predictions.append(out_item)
        if fw is not None:
            fw.write(json.dumps(self.predictions[-1]) + "\n")
        else:
            with open(self.logging_file, "a") as f:
                f.write(json.dumps(self.predictions[-1]) + "\n")

    def get_results(self):
        save_dir = os.path.dirname(self.output_file)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        cnt = 0
        pass_at_k = 0
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

        assert pass_at_k <= len(self.predictions)
        json.dump(self.predictions, open(self.output_file, "w"), indent=2)

        if len(self.predictions) == 0:
            metrics = {"acc": 0, "pass@k": 0, "maj@k": 0, "correct": 0, "total": 0}
        else:
            metrics = {"acc": cnt / len(self.predictions), "pass@k": pass_at_k / len(self.predictions), "correct": cnt, "total": len(self.predictions)}
            if len(acc_data_topic) > 0:
                for key in acc_data_topic:
                    metrics[f"acc_{key}"] = acc_data_topic[key] / cnt_data_topic[key]
        json.dump(metrics, open(self.output_file.replace(".json", ".metrics.json"), "w"), indent=2)
        return metrics, []
