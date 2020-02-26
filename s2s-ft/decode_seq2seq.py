"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import logging
import glob
import argparse
import math
from tqdm import tqdm, trange
import numpy as np
import torch
import random
import pickle

from transformers import BertTokenizer, RobertaTokenizer
from s2s_ft.modeling_decoding import BertForSeq2SeqDecoder
from transformers.tokenization_bert import whitespace_tokenize
import s2s_ft.s2s_loader as seq2seq_loader
from s2s_ft.utils import load_and_cache_examples
from s2s_ft.tokenization_unilm import UnilmTokenizer


class WhitespaceTokenizer(object):
    def tokenize(self, text):
        return whitespace_tokenize(text)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name", default='bert-base-cased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, roberta-base, "
                             "roberta-large, unilm-base-cased, unilm-large-cased.")
    parser.add_argument("--model_recover_path", default=None, type=str, required=True,
                        help="The file of fine-tuned pretraining model.")
    # tokenizer_name
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="tokenizer name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--ffn_type', default=0, type=int,
                        help="0: default mlp; 1: W((Wx+b) elem_prod x);")
    parser.add_argument('--num_qkv', default=0, type=int,
                        help="Number of different <Q,K,V>.")
    parser.add_argument('--seg_emb', action='store_true',
                        help="Using segment embedding for self-attention.")

    # decoding parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument("--input_file", type=str, help="Input file")
    parser.add_argument('--subset', type=int, default=0,
                        help="Decode a subset of the input dataset.")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--split", type=str, default="",
                        help="Data split (train/val/test).")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--new_pos_ids', action='store_true',
                        help="Use new position ids for LMs.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")

    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=1, type=int)
    parser.add_argument('--need_score_traces', action='store_true')
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--mode', default="s2s",
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--max_tgt_length', type=int, default=128,
                        help="maximum length of target sequence")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")

    args = parser.parse_args()

    if args.need_score_traces and args.beam_size <= 1:
        raise ValueError(
            "Score trace is only available for beam search with beam size > 1.")
    if args.max_tgt_length >= args.max_seq_length - 2:
        raise ValueError("Maximum tgt length exceeds max seq length - 2.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
    else:
        random_seed = random.randint(0, 10000)
        logger.info("Set random seed as: {}".format(random_seed))
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
    setattr(args, "is_roberta", args.model_name.startswith("roberta"))
    setattr(args, "no_segment_embedding", args.is_roberta)
    if not args.model_name.startswith("unilm1.2"):
        setattr(args, "new_segment_ids", args.model_name.startswith("unilm-") or args.model_name.startswith("unilm1-"))
    else:
        setattr(args, "new_segment_ids", False)
    
    if args.is_roberta:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        vocab = tokenizer.encoder
        args.model_name = args.model_name.replace("roberta", "bert") + "-cased"
    else:
        if args.model_name.startswith("bert"):
            tokenizer = BertTokenizer.from_pretrained(
                args.tokenizer_name if args.tokenizer_name else args.model_name,
                do_lower_case=args.do_lower_case)
        else:
            tokenizer = UnilmTokenizer.from_pretrained(
                args.tokenizer_name if args.tokenizer_name else args.model_name,
                do_lower_case=args.do_lower_case)
            args.model_name = 'bert-' + args.model_name.split('-', 1)[-1]

        vocab = tokenizer.vocab

    tokenizer.max_len = args.max_seq_length

    pair_num_relation = 0
    bi_uni_pipeline = []
    cls_token = '<s>' if args.is_roberta else '[CLS]'
    sep_token = '</s>' if args.is_roberta else '[SEP]'
    pad_token = '<pad>' if args.is_roberta else '[PAD]'
    mask_token = '<mask>' if args.is_roberta else '[MASK]'

    bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(
        list(vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length,
        max_tgt_length=args.max_tgt_length, new_segment_ids=args.new_segment_ids, mode="s2s", num_qkv=args.num_qkv,
        s2s_special_token=args.s2s_special_token, s2s_add_segment=args.s2s_add_segment,
        s2s_share_segment=args.s2s_share_segment, pos_shift=args.pos_shift,
        cls_token=cls_token, sep_token=sep_token, pad_token=pad_token))

    # Prepare model
    cls_num_labels = 2
    type_vocab_size = 6 + \
        (1 if args.s2s_add_segment else 0) if args.new_segment_ids else 2
    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
        [mask_token, sep_token, sep_token])
    forbid_ignore_set = None
    if args.forbid_ignore_word:
        w_list = []
        for w in args.forbid_ignore_word.split('|'):
            if w.startswith('[') and w.endswith(']'):
                w_list.append(w.upper())
            else:
                w_list.append(w)
        forbid_ignore_set = set(tokenizer.convert_tokens_to_ids(w_list))
    print(args.model_recover_path)
    found_checkpoint_flag = False
    for model_recover_path in glob.glob(args.model_recover_path.strip()):
        logger.info("***** Recover model: %s *****", model_recover_path)
        found_checkpoint_flag = True
        model_recover = torch.load(model_recover_path)
        model = BertForSeq2SeqDecoder.from_pretrained(
            args.model_name, state_dict=model_recover, num_labels=cls_num_labels, num_rel=pair_num_relation,
            type_vocab_size=type_vocab_size, task_idx=3, mask_word_id=mask_word_id, search_beam_size=args.beam_size,
            length_penalty=args.length_penalty, eos_id=eos_word_ids, sos_id=sos_word_id,
            forbid_duplicate_ngrams=args.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set,
            ngram_size=args.ngram_size, min_len=args.min_len, mode=args.mode,
            max_position_embeddings=args.max_seq_length, ffn_type=args.ffn_type, num_qkv=args.num_qkv,
            seg_emb=args.seg_emb, pos_shift=args.pos_shift, is_roberta=args.is_roberta,
            no_segment_embedding=args.no_segment_embedding,
        )
        del model_recover

        if args.fp16:
            model.half()
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        torch.cuda.empty_cache()
        model.eval()
        next_i = 0
        max_src_length = args.max_seq_length - 2 - args.max_tgt_length

        to_pred = load_and_cache_examples(
            args.input_file, tokenizer, local_rank=-1, 
            cached_features_file=None, shuffle=False)

        input_lines = []
        for line in to_pred:
            input_lines.append(tokenizer.convert_ids_to_tokens(line["source_ids"])[:max_src_length])
        if args.subset > 0:
            logger.info("Decoding subset: %d", args.subset)
            input_lines = input_lines[:args.subset]

        input_lines = sorted(list(enumerate(input_lines)),
                             key=lambda x: -len(x[1]))
        output_lines = [""] * len(input_lines)
        score_trace_list = [None] * len(input_lines)
        total_batch = math.ceil(len(input_lines) / args.batch_size)

        with tqdm(total=total_batch) as pbar:
            batch_count = 0
            first_batch = True
            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + args.batch_size]
                buf_id = [x[0] for x in _chunk]
                buf = [x[1] for x in _chunk]
                next_i += args.batch_size
                batch_count += 1
                max_a_len = max([len(x) for x in buf])
                instances = []
                for instance in [(x, max_a_len) for x in buf]:
                    for proc in bi_uni_pipeline:
                        instances.append(proc(instance))
                with torch.no_grad():
                    batch = seq2seq_loader.batch_list_to_batch_tensors(
                        instances)
                    batch = [
                        t.to(device) if t is not None else None for t in batch]
                    input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
                    traces = model(input_ids, token_type_ids,
                                   position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)
                    if args.beam_size > 1:
                        traces = {k: v.tolist() for k, v in traces.items()}
                        output_ids = traces['pred_seq']
                    else:
                        output_ids = traces.tolist()
                    for i in range(len(buf)):
                        w_ids = output_ids[i]
                        output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                        output_tokens = []
                        for t in output_buf:
                            if t in (sep_token, pad_token):
                                break
                            output_tokens.append(t)
                        if args.is_roberta:
                            output_sequence = tokenizer.convert_tokens_to_string(output_tokens)
                        else:
                            output_sequence = ' '.join(detokenize(output_tokens))
                        if '\n' in output_sequence:
                            output_sequence = " [X_SEP] ".join(output_sequence.split('\n'))
                        output_lines[buf_id[i]] = output_sequence
                        if first_batch or batch_count % 50 == 0:
                            logger.info("{} = {}".format(buf_id[i], output_sequence))
                        if args.need_score_traces:
                            score_trace_list[buf_id[i]] = {
                                'scores': traces['scores'][i], 'wids': traces['wids'][i], 'ptrs': traces['ptrs'][i]}
                pbar.update(1)
                first_batch = False
        if args.output_file:
            fn_out = args.output_file
        else:
            fn_out = model_recover_path+'.'+args.split
        with open(fn_out, "w", encoding="utf-8") as fout:
            for l in output_lines:
                fout.write(l)
                fout.write("\n")

        if args.need_score_traces:
            with open(fn_out + ".trace.pickle", "wb") as fout_trace:
                pickle.dump(
                    {"version": 0.0, "num_samples": len(input_lines)}, fout_trace)
                for x in score_trace_list:
                    pickle.dump(x, fout_trace)

    if not found_checkpoint_flag:
        logger.info("Not found the model checkpoint file!")


if __name__ == "__main__":
    main()
