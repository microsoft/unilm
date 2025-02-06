# coding=utf-8
#
# Copyright 2020 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
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

import inspect
import logging
import sys
import os

import hydra
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm

from general_util.logger import setting_logger
from general_util.training_utils import set_seed, load_and_cache_examples

logger: logging.Logger


def default_collate_fn(batch):
    return batch[0]


def run_inference(cfg: DictConfig, dataset):
    post_processor = hydra.utils.instantiate(cfg.post_process)

    # Eval!
    logger.info("***** Running inference through OpenAI API *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", 1)

    eval_dataloader = DataLoader(dataset,
                                 # sampler=eval_sampler,
                                 batch_size=1,
                                 collate_fn=default_collate_fn,
                                 num_workers=cfg.num_workers,
                                 pin_memory=True,
                                 prefetch_factor=cfg.prefetch_factor)

    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=cfg.local_rank not in [-1, 0], dynamic_ncols=True):
        if "meta_data" in batch:
            meta_data = batch.pop("meta_data")
        else:
            meta_data = []

        # outputs = model(**batch)
        outputs = batch
        post_processor(meta_data, outputs)

    sig = inspect.signature(post_processor.get_results)
    post_kwargs = {}
    if "output_dir" in list(sig.parameters.keys()):
        post_kwargs["output_dir"] = cfg.output_dir

    results, predictions = post_processor.get_results(**post_kwargs)
    logger.info(f"=================== Results =====================")
    for key, value in results.items():
        logger.info(f"{key}: {value}")

    return results


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    global logger
    logger = setting_logger("", local_rank=cfg.local_rank)
    logger.warning(f"CPU cores: {os.cpu_count()}")

    # Set seed
    set_seed(cfg)

    # model = hydra.utils.call(cfg.model)

    logger.info(cfg.test_file)
    dataset = load_and_cache_examples(cfg, None, _split="test")

    # Test
    results = run_inference(cfg, dataset)

    return results


if __name__ == "__main__":

    os.environ["HYDRA_FULL_ERROR"] = "1"

    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--"):])
        else:
            hydra_formatted_args.append(arg)
    sys.argv = hydra_formatted_args

    main()
