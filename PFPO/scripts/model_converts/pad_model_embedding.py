import argparse
import os
from collections import OrderedDict
from glob import glob

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers.models.llama import LlamaForCausalLM, LlamaConfig
from accelerate import init_empty_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--tp_size", type=int)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.input_dir)
    model.resize_token_embeddings(new_num_tokens=None, pad_to_multiple_of=args.tp_size)
    model.save_pretrained(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.input_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
