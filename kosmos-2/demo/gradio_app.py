#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""
import sys
sys.path.append( '.' )

import unilm

import ast
import logging
import math
import os
import sys
import time
import re
import random
from argparse import Namespace
from collections import namedtuple

import numpy as np
import torch

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output

import sentencepiece as spm
from torchvision import transforms
from PIL import Image

from draw_box import *
import gradio as gr


# store the image path for visualize
global_image_path = None
global_image_tensor = None
global_cnt = 0

# This is simple maximum entropy normalization performed in Inception paper
inception_normalize = transforms.Compose(
    [transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])]
)

def square_transform(size=224):
    return transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )

def split_string(string, separators):
    """
    Function to split a given string based on a list of separators.

    Args:
    string (str): The input string to be split.
    separators (list): A list of separators to be used for splitting the string.

    Returns:
    A list containing the split string with separators included.
    """
    pattern = "|".join(re.escape(separator) for separator in separators) 
    result = re.split(f'({pattern})', string)  
    return [elem for elem in result if elem] 

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.interactive")


Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints img_src_tokens img_gpt_input_mask")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")

def get_interactive_tokens_and_lengths(self, lines, encode_fn, tokenizer=None):
    """
    line format: [image]path<tab>text<tab>[image]path
    model input: `<s> <image> image hidden </image> My cat looking very dignified.</s>`
    """
    image_feature_length = self.args.image_feature_length
    bos_id = self.dictionary.bos()
    eos_id = self.dictionary.eos()
    boi_id = self.dictionary.index("<image>")
    eoi_id = self.dictionary.index("</image>")
    
    def convert_one_line(input_str):
        # TODO: input interleave image and text
        token = []
        img_src_token = []
        img_gpt_input_mask = []
        segments = input_str.split('<tab>')
        token.append(bos_id)
        img_gpt_input_mask.append(0)
        for i, segment in enumerate(segments):
            if segment.startswith('[image]'):
                image_path = segment[7:]
                # read image and transform to tensor
                image = Image.open(image_path).convert("RGB")
                # update the global_path
                global global_image_path
                global_image_path = image_path
                image_tensor = square_transform(self.args.input_resolution)(image)
                img_src_token.append(image_tensor)
                global global_image_tensor
                global_image_tensor = image_tensor
                token.extend([boi_id] + list(range(4, image_feature_length+4)) + [eoi_id])
                
                img_gpt_input_mask.extend([0] + [1] * image_feature_length + [0])
            else:
                special_tokens = [self.source_dictionary[idx] for idx in range(tokenizer.vocab_size(), 
                                                                               len(self.source_dictionary))]
                split_special_token_words = []
                split_resutls = split_string(segment, special_tokens)
                for string in split_resutls:
                    if string in special_tokens:
                        # print(f"dict-length({len(self.source_dictionary)}), substring {string} is a special token")
                        split_special_token_words.append(string)
                    else:
                        encode_tokens = tokenizer.encode(string, out_type=str)
                        # print(f"dict-length({len(self.source_dictionary)}), substring {string} is not a special token, tokenized into {encode_tokens}")
                        split_special_token_words.extend(encode_tokens)
                segment = ' '.join(split_special_token_words)
                
                text_tokens = self.source_dictionary.encode_line(
                    encode_fn(segment), add_if_not_exist=False
                ).tolist()
                
                text_tokens = text_tokens[:-1] # </s> in token
                token.extend(text_tokens)
                img_gpt_input_mask.extend([0] * (len(text_tokens))) # </s> in token
        token.append(eos_id)
        # img_gpt_input_mask = img_gpt_input_mask[:-1]
        assert len(token) == len(img_gpt_input_mask) + 1 
        token = torch.LongTensor(token)
        img_gpt_input_mask = torch.LongTensor(img_gpt_input_mask)
        img_src_token = torch.stack(img_src_token, dim=0)
        return token, img_src_token, img_gpt_input_mask
    
    tokens = []
    img_src_tokens = []
    img_gpt_input_masks = []
    for src_str in lines:
        token, img_src_token, img_gpt_input_mask = convert_one_line(src_str)
        tokens.append(token)
        img_src_tokens.append(img_src_token)
        img_gpt_input_masks.append(img_gpt_input_mask)
    lengths = [t.numel() for t in tokens]
    
    return tokens, lengths, img_src_tokens, img_gpt_input_masks


def make_batches(lines, cfg, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)

    if cfg.generation.constraints:
        # Strip (tab-delimited) contraints, if present, from input lines,
        # store them in batch_constraints
        batch_constraints = [list() for _ in lines]
        for i, line in enumerate(lines):
            if "\t" in line:
                lines[i], *batch_constraints[i] = line.split("\t")

        # Convert each List[str] to List[Tensor]
        for i, constraint_list in enumerate(batch_constraints):
            batch_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for constraint in constraint_list
            ]

    if cfg.generation.constraints:
        constraints_tensor = pack_constraints(batch_constraints)
    else:
        constraints_tensor = None

    tokenizer = spm.SentencePieceProcessor()
    if os.path.exists('data/sentencepiece.bpe.model'):
        tokenizer.Load('data/sentencepiece.bpe.model')
    else:
        tokenizer = None
    tokens, lengths, img_src_tokens, img_gpt_input_mask = get_interactive_tokens_and_lengths(task, lines, encode_fn, tokenizer)

    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_caption_inference(
            tokens, lengths, img_src_tokens, img_gpt_input_mask, constraints=constraints_tensor
        ),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        img_src_tokens = batch["net_input"]["img_src_tokens"]
        img_gpt_input_mask = batch["net_input"]["img_gpt_input_mask"]
        constraints = batch.get("constraints", None)

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            img_src_tokens=img_src_tokens,
            img_gpt_input_mask=img_gpt_input_mask,
            constraints=constraints,
        )


def main(cfg: FairseqConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    start_time = time.time()
    total_translate_time = 0

    utils.import_user_module(cfg.common)

    if cfg.interactive.buffer_size < 1:
        cfg.interactive.buffer_size = 1
    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.batch_size = 1

    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        not cfg.dataset.batch_size
        or cfg.dataset.batch_size <= cfg.interactive.buffer_size
    ), "--batch-size cannot be larger than --buffer-size"

    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Setup task, e.g., translation
    logger.info("Task: {}".format(cfg.task))
    task = tasks.setup_task(cfg.task)

    # Load ensemble
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    if cfg.generation.constraints:
        logger.warning(
            "NOTE: Constrained decoding currently assumes a shared subword vocabulary."
        )

    if cfg.interactive.buffer_size > 1:
        logger.info("Sentence buffer size: %s", cfg.interactive.buffer_size)
    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info("Type the input sentence and press return:")
    start_id = 0
    
    def generate_predictions(image_input, text_input, do_sample, sampling_topp, sampling_temperature):
        
        if do_sample:
            cfg.generation.sampling = True
            cfg.generation.sampling_topp = sampling_topp
            cfg.generation.temperature = sampling_temperature
            cfg.generation.beam = 1
        else:
            cfg.generation.sampling = False
            cfg.generation.sampling_topp = -1.0
            cfg.generation.temperature = 1.0
            cfg.generation.beam = 1
        generator = task.build_generator(models, cfg.generation)
            
        if image_input is None:
            user_image_path = None
        else:
            user_image_path = "/tmp/user_input_test_image.jpg"
            image_input.save(user_image_path)
        
        if text_input.lower() == 'brief':
            inputs = f"[image]{user_image_path}<tab><grounding>An image of"
        else:
            inputs = f"[image]{user_image_path}<tab><grounding>Describe this image in detail:"
        
        print("inputs", inputs)
        inputs = [inputs,]
        
        results = []
        for batch in make_batches(inputs, cfg, task, max_positions, encode_fn):
            bsz = batch.src_tokens.size(0)
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            img_src_tokens = batch.img_src_tokens
            img_gpt_input_mask = batch.img_gpt_input_mask
            constraints = batch.constraints
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
                if constraints is not None:
                    constraints = constraints.cuda()

            sample = {
                "net_input": {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths,
                    "img_src_tokens": img_src_tokens,
                    "img_gpt_input_mask": img_gpt_input_mask,
                },
            }
            translate_start_time = time.time()
            translations = task.inference_step(
                generator, models, sample, constraints=constraints
            )
            translate_time = time.time() - translate_start_time
            # total_translate_time += translate_time
            list_constraints = [[] for _ in range(bsz)]
            if cfg.generation.constraints:
                list_constraints = [unpack_constraints(c) for c in constraints]
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                constraints = list_constraints[i]
                results.append(
                    (
                        start_id + id,
                        src_tokens_i,
                        hypos,
                        {
                            "constraints": constraints,
                            "time": translate_time / len(translations),
                        },
                    )
                )

        global global_cnt
        global_cnt += 1
        # sort output to match input order
        for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
            src_str = ""
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
                print("S-{}\t{}".format(global_cnt, src_str))
                print("W-{}\t{:.3f}\tseconds".format(global_cnt, info["time"]))
                for constraint in info["constraints"]:
                    print(
                        "C-{}\t{}".format(
                            global_cnt,
                            tgt_dict.string(constraint, cfg.common_eval.post_process),
                        )
                    )

            # Process top predictions
            for hypo in hypos[: min(len(hypos), cfg.generation.nbest)]:
                hypo_tokens, hypo_str, alignment = post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=cfg.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                detok_hypo_str = decode_fn(hypo_str)
                
                # show the results on the image
                response_str = detok_hypo_str.split('</image>')[-1]
                if global_image_path is not None:
                    basename = os.path.basename(global_image_path).split('.')[0]
                    vis_image = visualize_results_on_image(global_image_path, response_str, task.args.location_bin_size, f"output/store_vis_results/show_box_on_{basename}.jpg", show=False)
                # if global_image_tensor is not None:
                #     basename = os.path.basename(global_image_path).split('.')[0]
                #     vis_image = visualize_results_on_image(global_image_tensor, response_str, task.args.location_bin_size, f"output/store_vis_results/show_box_on_{basename}.jpg", show=False)
                
                clean_response_str = re.sub('<[^>]*>', '', response_str) 
                clean_response_str = ' '.join(clean_response_str.split()).strip()
                
                score = hypo["score"] / math.log(2)  # convert to base 2
                # original hypothesis (after tokenization and BPE)
                print("H-{}\t{}\t{}".format(global_cnt, score, hypo_str))
                # detokenized hypothesis
                print("D-{}\t{}\t{}".format(global_cnt, score, detok_hypo_str))
                print(
                    "P-{}\t{}".format(
                        global_cnt,
                        " ".join(
                            map(
                                lambda x: "{:.4f}".format(x),
                                # convert from base e to base 2
                                hypo["positional_scores"].div_(math.log(2)).tolist(),
                            )
                        ),
                    )
                )
                
                if cfg.generation.print_alignment:
                    alignment_str = " ".join(
                        ["{}-{}".format(src, tgt) for src, tgt in alignment]
                    )
                    print("A-{}\t{}".format(global_cnt, alignment_str))
    
        # return vis_image, str(clean_response_str), str(response_str)
        return vis_image, mark_texts(response_str)
    
    term_of_use = """
    ### Terms of use  
    By using this model, users are required to agree to the following terms:  
    The model is intended for academic and research purposes. 
    The utilization of the model to create unsuitable material is strictly forbidden and not endorsed by this work. 
    The accountability for any improper or unacceptable application of the model rests exclusively with the individuals who generated such content. 
    
    ### License
    This project is licensed under the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct).
    """

    with gr.Blocks(title="Kosmos-2", theme=gr.themes.Base()).queue() as demo:
        gr.Markdown(("""
            # Kosmos-2: Grounding Multimodal Large Language Models to the World
            [[Paper]](https://arxiv.org/abs/2306.14824) [[Code]](https://github.com/microsoft/unilm/blob/master/kosmos-2)
            """))
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Test Image")  
                text_input = gr.Radio(["Brief", "Detailed"], label="Description Type", value="Brief")
                do_sample = gr.Checkbox(label="Enable Sampling", info="(Please enable it before adjusting sampling parameters below)", value=False)
                with gr.Accordion("Sampling parameters", open=False) as sampling_parameters:
                    sampling_topp = gr.Slider(minimum=0.1, maximum=1, step=0.01, value=0.9, label="Sampling: Top-P") 
                    sampling_temperature = gr.Slider(minimum=0.1, maximum=1, step=0.01, value=0.7, label="Sampling: Temperature") 

                run_button = gr.Button(label="Run", visible=True) 
                
            with gr.Column():
                image_output = gr.Image(type="pil")
                text_output1 = gr.HighlightedText(
                                    label="Generated Description", 
                                    combine_adjacent=False,
                                    show_legend=True,
                                ).style(color_map={"box": "red"})
                
        with gr.Row():
            with gr.Column():
                gr.Examples(examples=[  
                            ["demo/images/two_dogs.jpg", "Detailed", False],
                            ["demo/images/snowman.png", "Brief", False],
                            ["demo/images/man_ball.png", "Detailed", False],
                        ], inputs=[image_input, text_input, do_sample])
            with gr.Column():
                gr.Examples(examples=[  
                            ["demo/images/six_planes.png", "Brief", False],
                            ["demo/images/quadrocopter.jpg", "Brief", False],  
                            ["demo/images/carnaby_street.jpg", "Brief", False],  
                        ], inputs=[image_input, text_input, do_sample])
        gr.Markdown(term_of_use)
        
        run_button.click(fn=generate_predictions, 
                         inputs=[image_input, text_input, do_sample, sampling_topp, sampling_temperature],  
                         outputs=[image_output, text_output1],  
                         show_progress=True, queue=True)

    demo.launch(share=True)

# process the generated description for highlighting
def remove_special_fields(text):  
    return re.sub('<.*?>', '', text)  
  
def find_phrases(text):  
    phrases = re.finditer('<phrase>(.*?)</phrase>', text)  
    return [(match.group(1), match.start(1), match.end(1)) for match in phrases]  
  
def adjust_phrase_positions(phrases, text):  
    positions = []  
    for phrase, start, end in phrases:  
        adjusted_start = len(remove_special_fields(text[:start]))  
        adjusted_end = len(remove_special_fields(text[:end]))  
        positions.append((phrase, adjusted_start, adjusted_end))  
    return positions  
  
def mark_words(text, phrases):  
    marked_words = []  
  
    words = re.findall(r'\b\w+\b|[.,;?!:()"“”‘’\']', text)  
    word_indices = [match.start() for match in re.finditer(r'\b\w+\b|[.,;?!:()"“”‘’\']', text)]  
  
    for i, word in enumerate(words):  
        if any(start <= word_indices[i] < end for _, start, end in phrases):  
            marked_words.append((word, 'box'))  
        else:  
            marked_words.append((word, None))  
  
    return marked_words  

def merge_adjacent_words(marked_words):  
    merged_words = []  
    current_word, current_flag = marked_words[0]  
  
    for word, flag in marked_words[1:]:  
        if flag == current_flag:  
            current_word += " " + word  
        else:  
            merged_words.append((current_word, current_flag))  
            current_word = word  
            current_flag = flag  
  
    merged_words.append((current_word, current_flag))  
    return merged_words

def mark_texts(text):
    cleaned_text = remove_special_fields(text)      
    phrases = find_phrases(text)  
    adjusted_phrases = adjust_phrase_positions(phrases, text)      
    marked_words = mark_words(cleaned_text, adjusted_phrases)  
    merge_words = merge_adjacent_words(marked_words)
    return merge_words

# changed from fairseq.utils.py
def post_process_prediction(
    hypo_tokens,
    src_str,
    alignment,
    align_dict,
    tgt_dict,
    remove_bpe=None,
    extra_symbols_to_ignore=None,
):
    hypo_str = tgt_dict.string(
        hypo_tokens, remove_bpe, extra_symbols_to_ignore=extra_symbols_to_ignore
    )
    if align_dict is not None:
        hypo_str = utils.replace_unk(
            hypo_str, src_str, alignment, align_dict, tgt_dict.unk_string()
        )
    if align_dict is not None or remove_bpe is not None:
        # Convert back to tokens for evaluating with unk replacement or without BPE
        # Note that the dictionary can be modified inside the method.
        hypo_tokens = tgt_dict.encode_line(hypo_str, add_if_not_exist=False)
    return hypo_tokens, hypo_str, alignment

def cli_main():
    parser = options.get_interactive_generation_parser()
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(convert_namespace_to_omegaconf(args), main)


if __name__ == "__main__":    
    cli_main()