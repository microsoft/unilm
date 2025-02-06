# coding=utf-8
#
# Copyright 2023 Nanyang Technological University Fangkai Jiao
#
# Part of this code is based on the source code of Transformers
# (arXiv:1910.03771)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import inspect
import json
import logging
import os
import sys
from typing import List

import hydra
import torch
import vllm
from omegaconf import DictConfig
from tqdm import trange, tqdm
from transformers import PreTrainedTokenizer
from vllm import SamplingParams, RequestOutput

from general_util.logger import setting_logger
from general_util.training_utils import load_and_cache_examples

logger: logging.Logger

torch.backends.cuda.matmul.allow_tf32 = True


def evaluate(cfg: DictConfig, model: vllm.LLM, prefix="", _split="dev"):
    dataset = load_and_cache_examples(cfg, None, _split=_split)
    tokenizer: PreTrainedTokenizer = model.get_tokenizer()

    output_dir = getattr(cfg, "predict_dir", cfg.output_dir)

    if cfg.local_rank in [-1, 0] and not os.path.exists(os.path.join(output_dir, prefix)):
        os.makedirs(os.path.join(output_dir, prefix), exist_ok=True)

    post_processor = hydra.utils.instantiate(cfg.post_process) if "post_process" in cfg and cfg.post_process else None

    # Eval!
    torch.cuda.empty_cache()
    logger.info("***** Running evaluation {}.{} *****".format(_split, prefix))
    logger.info("  Num examples = %d", len(dataset))

    all_prompts = []
    all_meta_data = []

    for i in trange(len(dataset)):
        if cfg.local_rank != -1 and i % cfg.world_size != cfg.local_rank:
            continue
        inputs = dataset.api_getitem(i)
        if getattr(cfg, "apply_chat_template", False):
            all_prompts.append(tokenizer.apply_chat_template(conversation=[
                {"role": "user", "content": inputs.pop("text")}
            ],
                tokenize=False,
                add_generation_prompt=getattr(cfg, "add_generation_prompt", True)))
        else:
            all_prompts.append(inputs.pop("text"))
        all_meta_data.append(inputs.pop("meta_data"))

    sampling_params: SamplingParams = hydra.utils.instantiate(cfg.sampling_params)
    logger.warning(f"Sampling params: {sampling_params}")

    global_batch_size = getattr(cfg, "global_batch_size", len(all_prompts))
    if len(all_prompts) > 0:
        if any(hasattr(post_processor, tmp) for tmp in ["gather", "gather_object"]):
            kwargs = {
                "ddp": cfg.ddp_eval and cfg.local_rank != -1
            }
        else:
            kwargs = {}
        for i in trange(0, len(all_prompts), global_batch_size, desc="Batch inference"):
            batch_prompts = all_prompts[i:i + global_batch_size]
            batch_meta_data = all_meta_data[i:i + global_batch_size]
            outputs: List[RequestOutput] = model.generate(batch_prompts, sampling_params)
            if len(outputs) != len(batch_meta_data):
                logger.warning(f"outputs: {len(outputs)}, meta_data: {len(batch_meta_data)}")

            if hasattr(post_processor, "batch_call"):
                batch_outputs = [{"response": output} for output in outputs]
                post_processor.batch_call(batch_meta_data, batch_outputs, **kwargs)
            else:
                for output, meta_data in tqdm(zip(outputs, batch_meta_data), total=len(batch_meta_data), desc="Post-processing"):
                    output = {"response": output}
                    post_processor(meta_data, output, **kwargs)

    results = {}

    sig = inspect.signature(post_processor.get_results)
    post_kwargs = {}
    if "output_dir" in list(sig.parameters.keys()):
        post_kwargs["output_dir"] = os.path.join(output_dir, prefix)

    post_results, post_predictions = post_processor.get_results(**post_kwargs)
    results.update(post_results)
    metric_log = '\t'.join([f"{k}: {v}" for k, v in results.items()])
    predictions = post_predictions

    logger.info("****** Evaluation Results ******")
    logger.info(f"Global Steps: {prefix}")
    logger.info(metric_log)

    if len(predictions) > 0:
        if cfg.local_rank == -1:
            prediction_file = os.path.join(output_dir, prefix, "eval_predictions.json")
        else:
            prediction_file = os.path.join(output_dir, prefix, f"eval_predictions_rank{cfg.local_rank}.json")
        json.dump(predictions, open(prediction_file, "w"), indent=2)

    torch.cuda.empty_cache()

    return results


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))
    cfg.n_gpu = torch.cuda.device_count()
    cfg.device = device

    global logger
    logger = setting_logger(cfg.output_dir, local_rank=cfg.local_rank)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   cfg.local_rank, cfg.device, cfg.n_gpu, bool(cfg.local_rank != -1), cfg.fp16)
    logger.warning(f"CPU cores: {os.cpu_count()}")

    # Test
    results = {}

    checkpoints = [cfg.output_dir]
    if cfg.save_best:
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    # elif cfg.prediction_cfg.best_checkpoint and os.path.exists(cfg.prediction_cfg.best_checkpoint):
    #     checkpoints = [cfg.prediction_cfg.best_checkpoint]
    #     logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    elif cfg.eval_sub_path:
        checkpoints = list(sorted(list(set(
            os.path.dirname(c) for c in
            glob.glob(cfg.output_dir + f"/{cfg.eval_sub_path}/" + "pytorch_model*.bin", recursive=True)
        ))))
        if len(checkpoints) == 0:
            checkpoints = list(sorted(list(set(
                os.path.dirname(c) for c in
                glob.glob(cfg.output_dir + f"/{cfg.eval_sub_path}/" + "model*.safetensors", recursive=True)
            ))))
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    logger.info(" the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
        global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
        prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        split = "dev"

        model = vllm.LLM(model=checkpoint,
                         tensor_parallel_size=cfg.n_gpu,
                         swap_space=getattr(cfg, "swap_space", 32),
                         gpu_memory_utilization=float(getattr(os.environ, "gpu_memory_utilization", 0.95)),
                         load_format=getattr(cfg, "load_format", "auto"),
                         max_num_seqs=getattr(cfg, "max_num_seqs", 256),
                         seed=cfg.seed,
                         dtype="bfloat16" if cfg.fp16_bfloat16 else "float16",
                         distributed_executor_backend=getattr(cfg, "distributed_executor_backend", "ray"),
                         max_model_len=getattr(cfg, "max_model_len", None),
                         )

        if cfg.test_file:
            prefix = f'test' + (f'-{prefix}' if prefix != "" else "")
            split = "test"

        result = evaluate(cfg, model, prefix=prefix, _split=split)
        result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
        results.update(result)

        del model

    return results


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["WANDB__SERVICE_WAIT"] = "1200"
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--"):])
        else:
            hydra_formatted_args.append(arg)
    sys.argv = hydra_formatted_args
    print(sys.argv)
    main()
