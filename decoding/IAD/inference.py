import logging, os, sys
import time
import torch
from torch import Tensor
from typing import Dict, List, Optional
import copy
from tqdm import tqdm
from omegaconf import open_dict
import fairseq
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq import utils
from fairseq.data import data_utils
import argparse

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
def fairseq_generate(data_lines, args, models, task, batch_size, beam_size, device):
    # beam search | greedy decoding implemented by fairseq
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
def forward_decoder(model,
                    input_tokens,
                    encoder_out,
                    incremental_state,
                    parallel_forward_start_pos=None,
                    temperature=1.0,
                    use_log_softmax=True):
    decoder_out = model.decoder.forward(input_tokens,
                                        encoder_out=encoder_out,
                                        incremental_state=incremental_state,
                                        parallel_forward_start_pos=parallel_forward_start_pos)
    decoder_out_tuple = (decoder_out[0].div_(temperature), decoder_out[1])
    if use_log_softmax:
        # 1, len, vocab
        probs = model.get_normalized_probs(decoder_out_tuple, log_probs=True, sample=None)
    else:
        probs = decoder_out_tuple[0]
    # len
    pred_tokens = torch.argmax(probs, dim=-1).squeeze(0)
    return pred_tokens

@torch.no_grad()
def baseline_generate(data_lines, model, task, device, no_use_logsoft=False, max_len=200):
    # simplified greedy decoding
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    data_size = len(data_lines)
    all_results = []
    start = time.perf_counter()
    logger.info(f'Baseline generate')
    for start_idx in tqdm(range(0, data_size)):
        bpe_line = data_lines[start_idx]
        src_tokens = src_dict.encode_line(bpe_line, add_if_not_exist=False).long()
        net_input = {'src_tokens': src_tokens.unsqueeze(0).to(device),
                     'src_lengths': torch.LongTensor([src_tokens.numel()]).to(device)}
        encoder_out = model.encoder.forward_torchscript(net_input)
        incremental_state = torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]],
                                               torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}))
        tokens = [tgt_dict.eos()]

        for step in range(0, max_len):
            cur_input_tokens = torch.tensor([tokens]).to(device).long()
            # scalar
            pred_token = forward_decoder(model,
                                         cur_input_tokens,
                                         encoder_out,
                                         incremental_state,
                                         use_log_softmax=not no_use_logsoft).item()
            if pred_token == tgt_dict.eos():
                break
            else:
                tokens.append(pred_token)
        all_results.append(tgt_dict.string(tokens[1:]))
    delta = time.perf_counter() - start
    remove_bpe_results = [line.replace('@@ ', '') for line in all_results]
    return remove_bpe_results, delta

def construct_hash_sets(sent, min_gram=1, max_gram=3):
    hash_dict = {}
    for i in range(0, len(sent) - min_gram + 1):
        for j in range(min_gram, max_gram+1):
            if i + j <= len(sent):
                ngram = tuple(sent[i: i+j])
                if ngram not in hash_dict:
                    hash_dict[ngram] = []
                hash_dict[ngram].append(i+j)
    return hash_dict

def find_hash_sets(hash_set, tokens, min_gram=1, max_gram=3):
    for i in range(min_gram, max_gram+1):
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
def aggressive_generate(data_lines, model, task, device, no_use_logsoft=False, max_len=200):
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    encoder_state_ids = []
    for i in range(len(model.decoder.layers)):
        encoder_state_ids.append(model.decoder.layers[i].encoder_attn._incremental_state_id)
    data_size = len(data_lines)
    all_results = []
    start_time = time.perf_counter()
    for start_idx in tqdm(range(0, data_size)):
        bpe_line = data_lines[start_idx]
        src_tokens = src_dict.encode_line(bpe_line, add_if_not_exist=False).long()
        net_input = {'src_tokens': src_tokens.unsqueeze(0).to(device),
                     'src_lengths': torch.LongTensor([src_tokens.numel()]).to(device)}
        encoder_out = model.encoder.forward_torchscript(net_input)
        incremental_state = torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]],
                                               torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}))

        src_tokens_remove_eos_list = src_tokens[:-1].tolist()
        src_hash = construct_hash_sets(src_tokens_remove_eos_list)
        start = 0
        tokens = [tgt_dict.eos()]
        while start < len(src_tokens_remove_eos_list) and len(tokens) < max_len + 1:
            cur_span_input_tokens = torch.tensor([tokens + src_tokens_remove_eos_list[start:]]).to(device).long()
            pred_tokens = forward_decoder(model,
                                          cur_span_input_tokens,
                                          encoder_out,
                                          incremental_state,
                                          parallel_forward_start_pos=len(tokens) - 1,
                                          use_log_softmax=not no_use_logsoft)
            pred_judge = pred_tokens.cpu() == src_tokens[start:]
            if all(pred_judge):
                tokens += src_tokens[start:].tolist()
                break
            else:
                wrong_pos = pred_judge.tolist().index(False)
                start += wrong_pos
                tokens.extend(pred_tokens.cpu().tolist()[: wrong_pos + 1])
                cut_incremental_state(incremental_state, keep_len=len(tokens) - 1, encoder_state_ids=encoder_state_ids)
                cur_len = len(tokens)
                for step in range(cur_len, max_len + 1):
                    cur_input_tokens = torch.tensor([tokens]).to(device).long()
                    pred_token = forward_decoder(model,
                                                 cur_input_tokens,
                                                 encoder_out,
                                                 incremental_state,
                                                 use_log_softmax=not no_use_logsoft).item()
                    if pred_token == tgt_dict.eos():
                        start = len(src_tokens_remove_eos_list)
                        break
                    else:
                        tokens.append(pred_token)
                    find_end_idx = find_hash_sets(src_hash, tokens)
                    if find_end_idx != -1:
                        start = find_end_idx
                        if start < len(src_tokens_remove_eos_list):
                            break
        if len(tokens) > max_len + 1:
            tokens = tokens[:max_len + 1]
        all_results.append(tgt_dict.string(tokens[1:]))
    delta = time.perf_counter() - start_time
    remove_bpe_results = [line.replace('@@ ', '') for line in all_results]
    return remove_bpe_results, delta

@torch.no_grad()
def paper_aggressive_generate(data_lines, model, task, device, no_use_logsoft=False, max_len=200):
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    encoder_state_ids = []
    for i in range(len(model.decoder.layers)):
        encoder_state_ids.append(model.decoder.layers[i].encoder_attn._incremental_state_id)
    data_size = len(data_lines)
    all_results = []
    start_time = time.perf_counter()
    for start_idx in tqdm(range(0, data_size)):
        bpe_line = data_lines[start_idx]
        src_tokens = src_dict.encode_line(bpe_line, add_if_not_exist=False).long()
        net_input = {'src_tokens': src_tokens.unsqueeze(0).to(device),
                     'src_lengths': torch.LongTensor([src_tokens.numel()]).to(device)}
        encoder_out = model.encoder.forward_torchscript(net_input)
        incremental_state = torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]],
                                               torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}))

        src_tokens_remove_eos_list = src_tokens[:-1].tolist()
        src_hash = construct_hash_sets(src_tokens_remove_eos_list)

        src_tokens_add_pad_list = torch.tensor(src_tokens_remove_eos_list + [-1])  # [..., -1]

        tokens = [tgt_dict.eos()]
        while (len(tokens) == 1 or tokens[-1] != tgt_dict.eos()) and len(tokens) < max_len + 1:
            if len(tokens) == 1:
                find_end_idx = 0
            else:
                find_end_idx = find_hash_sets(src_hash, tokens)
            if find_end_idx != -1 and find_end_idx < len(src_tokens_remove_eos_list):
                cur_span_input_tokens = torch.tensor([tokens + src_tokens_remove_eos_list[find_end_idx:]]).to(device).long()
                pred_tokens = forward_decoder(model,
                                              cur_span_input_tokens,
                                              encoder_out,
                                              incremental_state,
                                              parallel_forward_start_pos=len(tokens) - 1,
                                              use_log_softmax=not no_use_logsoft)
                pred_judge = pred_tokens.cpu() == src_tokens_add_pad_list[find_end_idx:]
                wrong_pos = pred_judge.tolist().index(False)
                tokens.extend(pred_tokens.cpu().tolist()[: wrong_pos + 1])
                cut_incremental_state(incremental_state, keep_len=len(tokens) - 1, encoder_state_ids=encoder_state_ids)
            else:
                cur_input_tokens = torch.tensor([tokens]).to(device).long()
                pred_token = forward_decoder(model,
                                             cur_input_tokens,
                                             encoder_out,
                                             incremental_state,
                                             use_log_softmax=not no_use_logsoft).item()
                tokens.append(pred_token)
        if len(tokens) > max_len + 1:
            tokens = tokens[:max_len + 1]
        if tokens[-1] == tgt_dict.eos():
            tokens = tokens[:-1]
        all_results.append(tgt_dict.string(tokens[1:]))
    delta = time.perf_counter() - start_time
    remove_bpe_results = [line.replace('@@ ', '') for line in all_results]
    return remove_bpe_results, delta

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='path to model file, e.g., /to/path/checkpoint_best.pt')
    parser.add_argument('--bin-data', type=str, required=True,
                        help='directory containing src and tgt dictionaries')
    parser.add_argument('--input-path', type=str, required=True,
                        help='path to eval file, e.g., /to/path/conll14.bpe.txt')
    parser.add_argument('--output-path', type=str, default=None,
                        help='path to output file, e.g., /to/path/conll14.pred.txt')
    parser.add_argument('--batch', type=int, default=None,
                        help='batch size')
    parser.add_argument('--beam', type=int, default=5,
                        help='beam size')
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='greedy/one-by-one decoding')
    parser.add_argument('--aggressive', action='store_true', default=False,
                        help='aggressive decoding')
    parser.add_argument('--no_use_logsoft', action='store_true', default=False,
                        help='not use log_softmax when aggressive decoding')
    parser.add_argument('--block', type=int, default=None)
    parser.add_argument('--match', type=int, default=1)
    parser.add_argument('--cpu', action='store_true', default=False)
    cmd_args = parser.parse_args()

    cmd_args.checkpoint_path = os.path.expanduser(cmd_args.checkpoint_path)
    cmd_args.bin_data = os.path.expanduser(cmd_args.bin_data)
    cmd_args.input_path = os.path.expanduser(cmd_args.input_path)
    cmd_args.output_path = os.path.expanduser(cmd_args.output_path)

    models, args, task = load_model_ensemble_and_task(filenames=[cmd_args.checkpoint_path],
                                                    arg_overrides={'data': cmd_args.bin_data})
    if cmd_args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    model = models[0].to(device).eval()

    with open(cmd_args.input_path, 'r') as f:
        bpe_sents = [l.strip() for l in f.readlines()]

    if cmd_args.batch is not None:
        remove_bpe_results, delta = fairseq_generate(bpe_sents, args, models, task, cmd_args.batch, cmd_args.beam, device)
        logger.info(f'Fairseq generate batch {cmd_args.batch}, beam {cmd_args.beam}: {delta}')
    elif cmd_args.baseline:
        remove_bpe_results, delta = baseline_generate(bpe_sents, model, task, device, no_use_logsoft=cmd_args.no_use_logsoft)
        logger.info(f'Baseline generate: {delta}')
    elif cmd_args.aggressive:
        remove_bpe_results, delta = paper_aggressive_generate(bpe_sents, model, task, device, no_use_logsoft=cmd_args.no_use_logsoft)
        # remove_bpe_results, delta = aggressive_generate(bpe_sents, model, task, device, no_use_logsoft=cmd_args.no_use_logsoft)
        logger.info(f'Aggressive generate: {delta}')
    if cmd_args.output_path is not None:
        write_result(remove_bpe_results, cmd_args.output_path)
