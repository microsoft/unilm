from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import glob
import timeit

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
)

from markuplmft.models.markuplm import MarkupLMConfig, MarkupLMTokenizer, MarkupLMTokenizerFast, MarkupLMForQuestionAnswering

from utils import StrucDataset
from utils import (read_squad_examples, convert_examples_to_features, RawResult, write_predictions)
from utils_evaluate import EvalOpts, main as evaluate_on_squad

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    mp = "../../../../../results/markuplm-base"
    op = "./moli"
    config = MarkupLMConfig.from_pretrained(mp)
    logger.info("=====Config for model=====")
    logger.info(str(config))
    max_depth = config.max_depth
    tokenizer = MarkupLMTokenizer.from_pretrained(mp)
    model = MarkupLMForQuestionAnswering.from_pretrained(mp, config=config)

    tokenizer.save_pretrained(op)