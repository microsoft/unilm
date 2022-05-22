import os
import sys
import time
import torch
import logging
import argparse
import copy
from tqdm import tqdm
from torch import Tensor
from omegaconf import open_dict
from typing import Dict, Optional
from fairseq import utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s |  [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("inference")


def write_result(results, output_file):
    with open(output_file, 'w') as f:
        for line in results:
            f.write(line + '\n')


@torch.no_grad()
def forward_decoder(model, input_tokens, encoder_out, temperature=1.0, incremental_state=None,
                    parallel_forward_start_pos=None, use_log_softmax=False):
    decoder_out = model.decoder.forward(input_tokens,
                                        encoder_out=encoder_out,
                                        incremental_state=incremental_state,
                                        parallel_forward_start_pos=parallel_forward_start_pos)
    decoder_out_tuple = (decoder_out[0].div_(temperature), decoder_out[1])
    if use_log_softmax:
        probs = model.get_normalized_probs(decoder_out_tuple, log_probs=True, sample=None)
    else:
        probs = decoder_out_tuple[0]
    pred_tokens = torch.argmax(probs, dim=-1).squeeze(0)
    return pred_tokens


@torch.no_grad()
def fairseq_generate(data_lines, args, models, task, batch_size, beam_size, device):
    """beam search | greedy decoding implemented by fairseq"""
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    gen_args = copy.copy(args)
    with open_dict(gen_args):
        gen_args.beam = beam_size
    generator = task.build_generator(models, gen_args)
    data_size = len(data_lines)
    all_results = []
    logger.info(f'Fairseq generate batch {batch_size}, beam {beam_size}')
    start = time.perf_counter()
    for start_idx in tqdm(range(0, data_size, batch_size)):
        batch_lines = [line for line in data_lines[start_idx: min(start_idx + batch_size, data_size)]]
        batch_ids = [src_dict.encode_line(sentence, add_if_not_exist=False).long() for sentence in batch_lines]
        lengths = torch.LongTensor([t.numel() for t in batch_ids])
        batch_dataset = task.build_dataset_for_inference(batch_ids, lengths)
        batch_dataset.left_pad_source = True
        batch = batch_dataset.collater(batch_dataset)
        batch = utils.apply_to_sample(lambda t: t.to(device), batch)

        translations = generator.generate(models, batch, prefix_tokens=None)
        results = []
        for id, hypos in zip(batch["id"].tolist(), translations):
            results.append((id, hypos))
        batched_hypos = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]
        all_results.extend([tgt_dict.string(hypos[0]['tokens']) for hypos in batched_hypos])
    delta = time.perf_counter() - start
    remove_bpe_results = [line.replace('@@ ', '') for line in all_results]
    return remove_bpe_results, delta


@torch.no_grad()
def baseline_generate(data_lines, model, task, batch_size, device, no_use_logsoft=True, max_len=200):
    """batch Implementation"""
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    data_size = len(data_lines)
    all_results = []
    start = time.perf_counter()
    logger.info(f'Baseline generate')
    for start_idx in tqdm(range(0, data_size, batch_size)):
        batch_size = min(data_size - start_idx, batch_size)
        batch_lines = [line for line in data_lines[start_idx: start_idx + batch_size]]
        batch_ids = [src_dict.encode_line(sentence, add_if_not_exist=False).long() for sentence in batch_lines]
        lengths = torch.LongTensor([t.numel() for t in batch_ids])
        batch_dataset = task.build_dataset_for_inference(batch_ids, lengths)
        batch_dataset.left_pad_source = True
        batch = batch_dataset.collater(batch_dataset)
        batch = utils.apply_to_sample(lambda t: t.to(device), batch)
        net_input = batch['net_input']
        encoder_out = model.encoder.forward(net_input['src_tokens'], net_input['src_lengths'])
        incremental_state = torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]],
                                               torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}))

        batch_tokens = [[tgt_dict.eos()] for _ in range(batch_size)]
        finish_list = []
        for step in range(0, max_len):
            cur_input_tokens = torch.tensor(batch_tokens).to(device).long()
            pred_tokens = forward_decoder(model,
                                          cur_input_tokens,
                                          encoder_out,
                                          incremental_state,
                                          use_log_softmax=not no_use_logsoft,
                                          )
            for i, pred_tok in enumerate(pred_tokens):
                if len(batch_tokens[i]) == 1:
                    batch_tokens[i].append(pred_tok.item())
                else:
                    if batch_tokens[i][-1] != tgt_dict.eos():
                        batch_tokens[i].append(pred_tok.item())
                    else:
                        if i not in finish_list:
                            finish_list.append(i)
                        batch_tokens[i].append(tgt_dict.eos())
            if len(finish_list) == batch_size:
                break
        batch_tokens = [y for x, y in sorted(zip(batch['id'].cpu().tolist(), batch_tokens))]
        for tokens in batch_tokens:
            all_results.append(tgt_dict.string(tokens[1:]))
    remove_bpe_results = [line.replace('@@ ', '') for line in all_results]
    delta = time.perf_counter() - start
    return remove_bpe_results, delta


def construct_hash_sets(batch_sents, min_gram=1, max_gram=3):
    """batch Implementation"""
    batch_hash_dicts = []
    for sent in batch_sents:
        hash_dict = {}
        for i in range(0, len(sent) - min_gram + 1):
            for j in range(min_gram, max_gram + 1):
                if i + j <= len(sent):
                    ngram = tuple(sent[i: i + j])
                    if ngram not in hash_dict:
                        hash_dict[ngram] = []
                    hash_dict[ngram].append(i + j)
        batch_hash_dicts.append(hash_dict)
    return batch_hash_dicts


def find_hash_sets(hash_set, tokens, min_gram=1, max_gram=3):
    for i in range(min_gram, max_gram + 1):
        if len(tokens) < i:
            return -1
        ngram = tuple(tokens[-i:])
        if ngram not in hash_set:
            return -1
        if len(hash_set[ngram]) == 1:
            return hash_set[ngram][0]
    return -1


def cut_incremental_state(incremental_state, keep_len, encoder_state_ids):
    for n in incremental_state:
        if n[: n.index('.')] in encoder_state_ids:
            continue
        for k in incremental_state[n]:
            if incremental_state[n][k] is not None:
                if incremental_state[n][k].dim() == 4:
                    incremental_state[n][k] = incremental_state[n][k][:, :, :keep_len]
                elif incremental_state[n][k].dim() == 2:
                    incremental_state[n][k] = incremental_state[n][k][:, :keep_len]


@torch.no_grad()
def aggressive_generate(data_lines, model, task, batch_size, device, max_len=200):
    """batch Implementation"""
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    data_size = len(data_lines)
    all_results = []
    start_time = time.perf_counter()
    for start_idx in tqdm(range(0, data_size, batch_size)):
        batch_results = [[tgt_dict.eos()] for _ in range(batch_size)]
        batch_size = min(data_size - start_idx, batch_size)
        batch_lines = [line for line in data_lines[start_idx: start_idx + batch_size]]
        batch_ids = [src_dict.encode_line(sentence, add_if_not_exist=False).long() for sentence in batch_lines]
        lengths = torch.LongTensor([t.numel() for t in batch_ids])
        batch_dataset = task.build_dataset_for_inference(batch_ids, lengths)
        batch_dataset.left_pad_source = False
        batch = batch_dataset.collater(batch_dataset)
        batch = utils.apply_to_sample(lambda t: t.to(device), batch)
        net_input = batch['net_input']
        encoder_out = model.encoder.forward(net_input['src_tokens'], net_input['src_lengths'])
        src_tokens = net_input['src_tokens'].tolist()
        batch_tokens = [[tgt_dict.eos()] for _ in range(batch_size)]
        line_id = batch['id'].cpu().tolist()
        # remove padding, for hash construct
        batch_src_lines = [batch_ids[line_id[i]].cpu().tolist() for i in range(0, batch_size)]
        src_hash_lists = construct_hash_sets(batch_src_lines)
        finish_list = []
        at_list = []
        # pred token position
        start_list = [0] * batch_size
        # src token position
        src_pos_list = [0] * batch_size
        for step in range(0, max_len):
            # Aggressive Decoding at the first step
            if step == 0:
                cur_span_input_tokens = torch.tensor([[tgt_dict.eos()] + t for t in src_tokens]).to(device).long()
            else:
                # padding, 2 * max_len for boundary conditions
                pad_tokens = [([tgt_dict.eos()] + [tgt_dict.pad()] * max_len * 2) for _ in range(batch_size)]
                for i in range(batch_size):
                    index = max_len if max_len < len(batch_tokens[i]) else len(batch_tokens[i])
                    pad_tokens[i][:index] = batch_tokens[i][:index]
                cur_span_input_tokens = torch.tensor(pad_tokens).to(device)
                cur_span_input_tokens = cur_span_input_tokens[:, : cur_span_input_tokens.ne(tgt_dict.pad()).sum(1).max()]
            input_tokens_add = [t[1:] + [-1] for t in cur_span_input_tokens.cpu().tolist()]
            pred_tensor = forward_decoder(model, cur_span_input_tokens, encoder_out)
            pred_tokens = pred_tensor.cpu().tolist()
            if batch_size == 1:
                pred_tokens = [pred_tokens]
            for i, (input_token_add, pred_token) in enumerate(zip(input_tokens_add, pred_tokens)):
                if i not in finish_list:
                    # wrong pos is based on the src sent
                    wrong_pos = len(batch_src_lines[i][src_pos_list[i]:])
                    for j, (inp, pred) in enumerate(zip(input_token_add[start_list[i]:], pred_token[start_list[i]:])):
                        if inp != pred:
                            wrong_pos = j
                            break
                    if step == 0:
                        src_pos_list[i] += wrong_pos
                        batch_tokens[i].extend(pred_token[start_list[i]: start_list[i] + wrong_pos])
                    if (batch_tokens[i][-1] == tgt_dict.eos() and len(batch_tokens[i]) != 1
                        and wrong_pos >= len(batch_src_lines[i][src_pos_list[i]:])) or start_list[i] > max_len:
                        finish_list.append(i)
                        if len(batch_tokens[i]) > max_len + 1:
                            batch_tokens[i] = batch_tokens[i][:max_len + 1]
                        batch_results[i] = batch_tokens[i]
                    else:
                        if i not in at_list:
                            # greedy decoding
                            batch_tokens[i] = batch_tokens[i][: start_list[i] + wrong_pos + 1]
                            batch_tokens[i].append(pred_token[start_list[i] + wrong_pos])
                            start_list[i] = start_list[i] + wrong_pos + 1
                            at_list.append(i)
                        else:
                            batch_tokens[i].append(pred_token[start_list[i]])
                            start_list[i] += 1
                            find_end_idx = find_hash_sets(src_hash_lists[i], batch_tokens[i])
                            if find_end_idx != -1:
                                start_list[i] = len(batch_tokens[i]) - 1
                                src_pos_list[i] = find_end_idx
                                batch_tokens[i] = batch_tokens[i] + batch_src_lines[i][src_pos_list[i]:]
                                at_list.remove(i)
            if len(finish_list) == batch_size:
                break
        batch_results = [y for x, y in sorted(zip(line_id, batch_results))]
        for tokens in batch_results:
            all_results.append(tgt_dict.string(tokens[1:]))
    delta = time.perf_counter() - start_time
    remove_bpe_results = [line.replace('@@ ', '') for line in all_results]
    return remove_bpe_results, delta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='path to model file, e.g., /to/path/checkpoint_best.pt')
    parser.add_argument('--bin-data', type=str, default=None,
                        help='directory containing src and tgt dictionaries')
    parser.add_argument('--input-path', type=str, default=None,
                        help='path to eval file, e.g., /to/path/conll14.bpe.txt')
    parser.add_argument('--output-path', type=str, default=None,
                        help='path to output file, e.g., /to/path/conll14.pred.txt')
    parser.add_argument('--batch', type=int, default=10,
                        help='batch size')
    parser.add_argument('--beam', type=int, default=1,
                        help='beam size')
    parser.add_argument('--fairseq', action='store_true', default=False,
                        help='fairseq decoding')
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='greedy/batch decoding')
    parser.add_argument('--aggressive', action='store_true', default=False,
                        help='aggressive decoding')
    parser.add_argument('--block', type=int, default=None)
    parser.add_argument('--match', type=int, default=1)
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--fp16', action='store_true', default=False)
    cmd_args = parser.parse_args()

    cmd_args.checkpoint_path = os.path.expanduser(cmd_args.checkpoint_path)
    cmd_args.bin_data = os.path.expanduser(cmd_args.bin_data)
    cmd_args.input_path = os.path.expanduser(cmd_args.input_path)
    cmd_args.output_path = os.path.expanduser(cmd_args.output_path)

    models, args, task = load_model_ensemble_and_task(filenames=[cmd_args.checkpoint_path],
                                                      arg_overrides={'data': cmd_args.bin_data})

    device = torch.device('cuda')
    model = models[0].to(device).eval()
    if cmd_args.fp16:
        logging.info("fp16 enabled!")
        model.half()

    with open(cmd_args.input_path, 'r') as f:
        bpe_sents = [l.strip() for l in f.readlines()]

    remove_bpe_results = None
    if cmd_args.fairseq:
        remove_bpe_results, delta = fairseq_generate(bpe_sents, args, models, task, cmd_args.batch, cmd_args.beam,
                                                     device)
        logger.info(f'Fairseq generate batch {cmd_args.batch}, beam {cmd_args.beam}: {delta}')
    elif cmd_args.baseline:
        remove_bpe_results, delta = baseline_generate(bpe_sents, model, task, cmd_args.batch, device)
        logger.info(f'Baseline generate: {delta}')
    elif cmd_args.aggressive:
        remove_bpe_results, delta = aggressive_generate(bpe_sents, model, task, cmd_args.batch, device)
        logger.info(f'Aggressive generate: {delta}')

    if cmd_args.output_path is not None:
        write_result(remove_bpe_results, cmd_args.output_path)
