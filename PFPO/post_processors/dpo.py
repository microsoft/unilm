import json
import json
import os
from typing import Dict, Any

import numpy as np
import torch
from torch import distributed as dist

from post_processors.dist_mixin import DistGatherMixin
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class DPOEvalPostProcessor(DistGatherMixin):
    def __init__(self):
        super().__init__()
        self.predictions = []
        self.chosen_rewards = []
        self.rejected_rewards = []
        self.losses = []

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        index = meta_data["index"]
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        inputs = meta_data["prompt"]
        chosen = meta_data["chosen"]
        rejected = meta_data["reject"]

        chosen_rewards = batch_model_outputs["chosen_reward"].item()
        rejected_rewards = batch_model_outputs["rejected_reward"].item()
        loss = batch_model_outputs["loss"].item()

        if ddp:
            obj = [inputs, chosen, rejected, index, chosen_rewards, rejected_rewards, loss]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                inputs = []
                chosen = []
                rejected = []
                index = []
                chosen_rewards = []
                rejected_rewards = []
                loss = []
                for item in gather_res:
                    inputs.extend(item[0])
                    chosen.extend(item[1])
                    rejected.extend(item[2])
                    index.extend(item[3])
                    chosen_rewards.append(item[4])
                    rejected_rewards.append(item[5])
                    loss.append(item[6])

        self.predictions.extend([{
            "input": input,
            "chosen": chosen,
            "rejected": rejected,
            "index": index,
        } for input, chosen, rejected, index in zip(inputs, chosen, rejected, index)])
        self.chosen_rewards.append(chosen_rewards)
        self.rejected_rewards.append(rejected_rewards)
        self.losses.append(loss)

    def get_results(self, output_dir: str):
        # output_file = os.path.join(output_dir, "eval_predictions.npy")
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"eval_predictions_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "eval_predictions.json")

        self.predictions = sorted(self.predictions, key=lambda x: x["index"])

        avg_loss = np.mean(self.losses).item()
        avg_chosen_reward = np.mean(self.chosen_rewards).item()
        avg_rejected_reward = np.mean(self.rejected_rewards).item()

        metrics = {
            "loss": avg_loss,
            "chosen_reward": avg_chosen_reward,
            "rejected_reward": avg_rejected_reward,
        }

        json.dump(self.predictions, open(output_file, "w"), indent=2, ensure_ascii=False)
        json.dump(metrics, open(output_file.replace(".json", ".metrics.json"), "w"), indent=2, ensure_ascii=False)

        return metrics, self.predictions


class DPORewardPostProcessor(DistGatherMixin):
    def __init__(self):
        super().__init__()
        self.predictions = []
        self.losses = []

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        index = meta_data["index"]
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        inputs = meta_data["prompt"]
        chosen = meta_data["chosen"]
        rejected = meta_data["reject"]

        chosen_rewards = batch_model_outputs["batch_chosen_reward"].tolist()
        rejected_rewards = batch_model_outputs["batch_rejected_reward"].tolist()
        loss = batch_model_outputs["loss"].item()

        if ddp:
            obj = [inputs, chosen, rejected, index, chosen_rewards, rejected_rewards, loss]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                inputs = []
                chosen = []
                rejected = []
                index = []
                chosen_rewards = []
                rejected_rewards = []
                loss = []
                for item in gather_res:
                    inputs.extend(item[0])
                    chosen.extend(item[1])
                    rejected.extend(item[2])
                    index.extend(item[3])
                    chosen_rewards.extend(item[4])
                    rejected_rewards.extend(item[5])
                    loss.append(item[6])

        self.predictions.extend([{
            "input": prompt,
            "chosen": ch,
            "rejected": rej,
            "index": i,
            "chosen_reward": chosen_r,
            "rejected_reward": rejected_r,
        } for prompt, ch, rej, chosen_r, rejected_r, i in zip(inputs, chosen, rejected, chosen_rewards, rejected_rewards, index)])
        self.losses.append(loss)

    def get_results(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"eval_predictions_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "eval_predictions.json")

        self.predictions = sorted(self.predictions, key=lambda x: x["index"])

        acc = np.mean([x["chosen_reward"] > x["rejected_reward"] for x in self.predictions]).item()

        metrics = {
            "acc": acc,
        }

        json.dump(self.predictions, open(output_file, "w"), indent=2, ensure_ascii=False)
        json.dump(metrics, open(output_file.replace(".json", ".metrics.json"), "w"), indent=2, ensure_ascii=False)

        return metrics, self.predictions


class ResponseClsPostProcessor(DistGatherMixin):
    def __init__(self):
        super().__init__()
        self.predictions = []

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        index = meta_data["index"]
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        inputs = meta_data["prompt"]
        responses = meta_data["response"]
        labels = meta_data["label"]

        logits = batch_model_outputs["logits"].tolist()

        if ddp:
            obj = [inputs, index, responses, logits, labels]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                inputs = []
                index = []
                responses = []
                logits = []
                labels = []
                for item in gather_res:
                    inputs.extend(item[0])
                    index.extend(item[1])
                    responses.extend(item[2])
                    logits.extend(item[3])
                    labels.extend(item[4])

        self.predictions.extend([{
            "input": prompt,
            "index": i,
            "response": resp,
            "logits": logit,
            "label": label,
        } for prompt, i, resp, logit, label in zip(inputs, index, responses, logits, labels)])

    def get_results(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"eval_predictions_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "eval_predictions.json")

        self.predictions = sorted(self.predictions, key=lambda x: x["index"])

        pred = [x["logits"] for x in self.predictions]
        pred = np.argmax(pred, axis=1).tolist()
        labels = [x["label"] for x in self.predictions]
        acc = np.mean([x == y for x, y in zip(pred, labels)]).item()

        metrics = {
            "acc": acc,
        }

        json.dump(self.predictions, open(output_file, "w"), indent=2, ensure_ascii=False)
        json.dump(metrics, open(output_file.replace(".json", ".metrics.json"), "w"), indent=2, ensure_ascii=False)

        return metrics, self.predictions


def process_response(response: str):
    lines = response.split("\n")
    lines = list(filter(lambda x: x[1].startswith("Thought ") or x[1].startswith("Action ") or x[1].startswith("Observation "), enumerate(lines)))
    return lines


def process_response_v2(response: str):
    lines = response.split("\n")
    outputs = []
    for line_id, line in enumerate(lines):

        if line.startswith("Thought ") or line.startswith("Action ") or line.startswith("Observation "):
            outputs.append({
                "text": line,
                "type": "text",
                "line_id": line_id,
            })
        elif not line.strip():
            outputs.append({
                "text": line,
                "type": "space",
                "line_id": line_id,
            })
        else:
            outputs.append({
                "text": line,
                "type": "continue",
                "line_id": line_id,
            })

    compose_outputs = []
    for item in outputs:
        if item["type"] == "text":
            compose_outputs.append((item["line_id"], item["text"]))
        elif item["type"] == "space":
            if len(compose_outputs):
                tmp = compose_outputs[-1]
                new_line_text = "\n".join([tmp[1], item["text"]])
                compose_outputs[-1] = (tmp[0], new_line_text)
        else:
            if len(compose_outputs):
                tmp = compose_outputs[-1]
                new_line_text = "\n".join([tmp[1], item["text"]])
                compose_outputs[-1] = (item["line_id"], new_line_text)
            else:
                compose_outputs.append((item["line_id"], item["text"]))

    outputs = []
    for item in compose_outputs:
        if item[1].startswith("Thought "):
            content = item[1][len("Thought "):]
            content = content.strip()
            if len(content) >= 5 or item[1].startswith("Thought 1:"):  # FIXED: Hack for LogiQA-v2 reward model evaluation, where the responses are not cleaned. @2024/01/18.
                outputs.append(item)
        elif item[1].startswith("Action "):
            content = item[1][len("Action "):]
            content = content.strip()
            if len(content) >= 5:
                outputs.append(item)
        elif item[1].startswith("Observation "):
            content = item[1][len("Observation "):]
            content = content.strip()
            if len(content) >= 5:
                outputs.append(item)
        else:
            # logger.warning(f"Warning: Unknown line: {item[1]}")
            if len(item[1]) >= 5:
                outputs.append(item)

    return outputs


class ResponseProcessRewardPostProcessor(DistGatherMixin):
    def __init__(self, reduction: str = "product", prob_labels: str = "(2,3)"):
        """
        :param reduction: "product|min"
        """
        super().__init__()
        self.predictions = []
        self.reduction = reduction
        self.prob_labels = eval(prob_labels)
        logger.info(f"prob_labels: {self.prob_labels}")

    def logit2prob(self, logits):
        probs = torch.softmax(logits, dim=-1)
        probs = probs[:, self.prob_labels].sum(dim=-1)
        return probs

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        index = meta_data["index"]
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        inputs = meta_data["prompt"]
        responses = meta_data["response"]
        ending_positions = meta_data["ending"]

        logits = batch_model_outputs["logits"].tolist()

        for i, endings in enumerate(ending_positions):
            # tmp = len(process_response("Thought 1: " + responses[i]))
            tmp = inputs[i] + responses[i]
            tmp = len(process_response_v2(tmp[tmp.find("Thought 1:"):]))  # FIXED: @2024/01/06 for ReClor.
            assert len(endings) == tmp, (len(endings), tmp, endings, responses[i], inputs[i])

        ending_logits = []
        assert len(ending_positions) == len(logits)
        for endings, seq_logits in zip(ending_positions, logits):
            try:
                ending_logits.append([seq_logits[e] for e in endings])
            except IndexError:
                print(endings)
                print(len(seq_logits))
                raise

        if ddp:
            obj = [inputs, index, responses, ending_logits]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                inputs = []
                index = []
                responses = []
                ending_logits = []
                for item in gather_res:
                    inputs.extend(item[0])
                    index.extend(item[1])
                    responses.extend(item[2])
                    ending_logits.extend(item[3])

        self.predictions.extend([{
            "input": prompt,
            "index": i,
            "response": resp,
            "ending_logits": logits,
        } for prompt, i, resp, logits in zip(inputs, index, responses, ending_logits)])

    def get_results(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"eval_predictions_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "eval_predictions.json")

        self.predictions = sorted(self.predictions, key=lambda x: x["index"])

        for pred in self.predictions:
            logits = torch.tensor(pred["ending_logits"])
            probs = self.logit2prob(logits)
            if self.reduction == "product":
                pred["reward"] = probs.prod().item()
            elif self.reduction == "min":
                pred["reward"] = probs.min().item()
            else:
                raise ValueError(f"Unknown reduction: {self.reduction}")

        json.dump(self.predictions, open(output_file, "w"), indent=2, ensure_ascii=False)

        return {}, self.predictions


class ResponseProcessRewardPostProcessorV2(DistGatherMixin):
    def __init__(self, reduction: str = "product", prob_labels: str = "(2,3)"):
        """
        :param reduction: "product|min"
        """
        super().__init__()
        self.predictions = []
        self.reduction = reduction
        self.prob_labels = eval(prob_labels)
        logger.info(f"prob_labels: {self.prob_labels}")

    def logit2prob(self, logits):
        probs = torch.softmax(logits, dim=-1)
        probs = probs[:, self.prob_labels].sum(dim=-1)
        return probs

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        index = meta_data["index"]
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        inputs = meta_data["prompt"]
        responses = meta_data["response"]
        ending_positions = meta_data["ending"]
        types = meta_data["type"]

        logits = batch_model_outputs["logits"].tolist()

        # Comment the following for dataset not using ReAct format.
        # for i, endings in enumerate(ending_positions):
        #     tmp = inputs[i] + responses[i]
        #     tmp = len(process_response_v2(tmp[tmp.find("Thought 1:"):]))  # FIXED: @2024/01/06 for ReClor.
        #     assert len(endings) == tmp, (len(endings), tmp, endings, responses[i], inputs[i])

        ending_logits = []
        assert len(ending_positions) == len(logits)
        for endings, seq_logits in zip(ending_positions, logits):
            try:
                ending_logits.append([seq_logits[e] for e in endings])
            except IndexError:
                print(endings)
                print(len(seq_logits))
                raise

        if ddp:
            obj = [inputs, index, responses, ending_logits, types]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                inputs = []
                index = []
                responses = []
                ending_logits = []
                types = []
                for item in gather_res:
                    inputs.extend(item[0])
                    index.extend(item[1])
                    responses.extend(item[2])
                    ending_logits.extend(item[3])
                    types.extend(item[4])

        self.predictions.extend([{
            "input": prompt,
            "index": i,
            "response": resp,
            "ending_logits": logits,
            "step_types": step_types,
        } for prompt, i, resp, logits, step_types in zip(inputs, index, responses, ending_logits, types)])

    def get_results(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"eval_predictions_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "eval_predictions.json")

        self.predictions = sorted(self.predictions, key=lambda x: x["index"])

        for pred in self.predictions:
            if len(pred["ending_logits"]) == 0:
                pred["reward"] = 0.0
                continue
            logits = torch.tensor(pred["ending_logits"])
            probs = self.logit2prob(logits)
            if self.reduction == "product":
                pred["reward"] = probs.prod().item()
            elif self.reduction == "min":
                pred["reward"] = probs.min().item()
            else:
                raise ValueError(f"Unknown reduction: {self.reduction}")

        json.dump(self.predictions, open(output_file, "w"), indent=2, ensure_ascii=False)

        return {}, self.predictions


class DPORewardSinglePostProcessor(DistGatherMixin):
    def __init__(self):
        super().__init__()
        self.predictions = []

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        index = meta_data["index"]
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        inputs = meta_data["prompt"]
        responses = meta_data["response"]

        rewards = batch_model_outputs["batch_chosen_reward"].tolist()

        if ddp:
            obj = [inputs, responses, index, rewards]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                inputs = []
                responses = []
                index = []
                rewards = []
                for item in gather_res:
                    inputs.extend(item[0])
                    responses.extend(item[1])
                    index.extend(item[2])
                    rewards.extend(item[3])

        self.predictions.extend([{
            "input": prompt,
            "response": resp,
            "index": i,
            "reward": r,
        } for prompt, resp, r, i in zip(inputs, responses, rewards, index)])

    def get_results(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if dist.is_initialized():
            dist.barrier()

        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"eval_predictions_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "eval_predictions.json")

        json.dump(self.predictions, open(output_file, "w"), indent=2, ensure_ascii=False)

        return {}, self.predictions
