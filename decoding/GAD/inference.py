import os
import sys
import time
import logging
from tqdm import tqdm

import torch
from fairseq import utils, tasks, options
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

from torch import Tensor
from typing import Dict, List, Optional

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
def fairseq_generate(data_lines, cfg, models, task, batch_size, device):
    # fairseq original decoding implementation
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    generator = task.build_generator(models, cfg.generation)
    data_size = len(data_lines)
    all_results = []
    logger.info(f'Fairseq generate batch {batch_size}')
    start = time.perf_counter()
    for start_idx in tqdm(range(0, data_size, batch_size)):
        batch_lines = [line for line in data_lines[start_idx: min(start_idx + batch_size, data_size)]]
        batch_ids = [src_dict.encode_line(sentence, add_if_not_exist=False).long() for sentence in batch_lines]
        lengths = torch.LongTensor([t.numel() for t in batch_ids])
        batch_dataset = task.build_dataset_for_inference(batch_ids, lengths)
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
def baseline_forward_decoder(model,
                             input_tokens,
                             encoder_out: Dict[str, List[Tensor]],
                             incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
                             parallel_forward_start_pos=None,
                             temperature: float = 1.0):
    decoder_out = model.decoder.forward(input_tokens,
                                        encoder_out=encoder_out,
                                        incremental_state=incremental_state,
                                        parallel_forward_start_pos=parallel_forward_start_pos)
    decoder_out_tuple = (decoder_out[0].div_(temperature), decoder_out[1])
    pred_tokens = torch.argmax(decoder_out_tuple[0], dim=-1).squeeze(0)
    return pred_tokens


@torch.no_grad()
def baseline_generate(data_lines, model, task, batch_size, device, max_len=200):
    # simplified AR greedy decoding
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
        batch_dataset.left_pad_source = False
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
            pred_tokens = baseline_forward_decoder(model,
                                                   cur_input_tokens,
                                                   encoder_out,
                                                   incremental_state=incremental_state)
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


@torch.no_grad()
def forward_decoder(model, input_tokens, encoder_out, incremental_state=None,
                    parallel_forward_start_pos=None, temperature=1.0, beta=1, tau=0.0):
    decoder_out = model.decoder.forward(input_tokens,
                                        encoder_out=encoder_out,
                                        incremental_state=incremental_state,
                                        parallel_forward_start_pos=parallel_forward_start_pos)
    decoder_out_tuple = (decoder_out[0].div_(temperature), decoder_out[1])
    topk_scores, indexes = torch.topk(decoder_out_tuple[0], beta, dim=-1)
    topk_scores_list = topk_scores.tolist()
    indexes_list = indexes.tolist()
    for i in range(indexes.size(0)):
        for j in range(indexes.size(1)):
            for k, s in enumerate(topk_scores_list[i][j]):
                if topk_scores_list[i][j][0] - s > tau:
                    indexes_list[i][j][k] = -1
    return indexes_list


def gad_generate(data_lines, model, AR_model, task, block_size, batch_size, device, beta=1, tau=0, max_len=200):
    # Generalized Aggressive Decoding
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    data_size = len(data_lines)
    all_results = []
    logger.info(f'GAD generate')
    start = time.perf_counter()
    for start_idx in tqdm(range(0, data_size, batch_size)):
        batch_size = min(data_size - start_idx, batch_size)
        batch_lines = [line for line in data_lines[start_idx: start_idx + batch_size]]
        batch_ids = [src_dict.encode_line(sentence, add_if_not_exist=False).long() for sentence in batch_lines]
        lengths = torch.LongTensor([t.numel() for t in batch_ids])
        batch_dataset = task.build_dataset_for_inference(batch_ids, lengths)
        batch_dataset.left_pad_source = False
        batch = batch_dataset.collater(batch_dataset)
        batch = utils.apply_to_sample(lambda t: t.to(device), batch)
        net_input = batch['net_input']
        AR_encoder_out = AR_model.encoder.forward(net_input['src_tokens'], net_input['src_lengths'])
        encoder_out = model.encoder.forward(net_input['src_tokens'], net_input['src_lengths'])
        sentences = [[tgt_dict.eos()] for _ in range(batch_size)]
        prev_output_tokens = [[tgt_dict.unk()] * block_size for _ in range(batch_size)]
        start_pos_list = [0] * batch_size
        finish_list = []
        for step in range(0, max_len):
            prev_output_tokens, start_pos_list = gad_forward(start_pos_list, block_size, batch_size,
                                                             tgt_dict, prev_output_tokens,
                                                             encoder_out, AR_encoder_out, model, AR_model, beta, tau)
            for i, start_pos in enumerate(start_pos_list):
                if i not in finish_list:
                    if start_pos == -1:
                        finish_list.append(i)
                        sentences[i] = prev_output_tokens[i]

            if len(finish_list) == batch_size:
                break
        batch_sents = [y for x, y in sorted(zip(batch['id'].cpu().tolist(), sentences))]
        for s in batch_sents:
            all_results.append(tgt_dict.string(s))
    remove_bpe_results = [line.replace('@@ ', '') for line in all_results]
    delta = time.perf_counter() - start
    return remove_bpe_results, delta


def gad_forward(start_pos_list, block_size, batch_size, tgt_dict, prev_output_tokens,
                encoder_out, AR_encoder_out, model, AR_model, beta, tau, max_len=200):
    pad_tokens = [[tgt_dict.pad()] * (max_len + block_size) for _ in range(batch_size)]
    for i in range(batch_size):
        pad_tokens[i][:len(prev_output_tokens[i])] = prev_output_tokens[i]
    output_tokens = torch.tensor(pad_tokens).to(device)
    output_tokens = output_tokens[:, : output_tokens.ne(tgt_dict.pad()).sum(1).max()]

    _, tensor_tokens = model.decoder(
        normalize=False,
        prev_output_tokens=output_tokens,
        encoder_out=encoder_out,
    ).max(-1)

    _tokens = tensor_tokens.tolist()
    for i, start_pos in enumerate(start_pos_list):
        if start_pos_list[i] != -1:
            output_tokens[i, start_pos:start_pos + block_size] = tensor_tokens[i, start_pos:start_pos + block_size]
            prev_output_tokens[i][start_pos:start_pos + block_size] = _tokens[i][start_pos:start_pos + block_size]

    append_eos = torch.tensor([[tgt_dict.eos()] for _ in range(batch_size)]).to(device)
    cur_span_input_tokens = torch.cat((append_eos, output_tokens), dim=-1)

    AR_verify_tokens = forward_decoder(AR_model, cur_span_input_tokens, AR_encoder_out, beta=beta, tau=tau)

    next_output_tokens = prev_output_tokens.copy()
    for i in range(batch_size):
        if start_pos_list[i] != -1:
            bifurcation = block_size
            for j, (token, AR_verify_token) in enumerate(
                    zip(prev_output_tokens[i][start_pos_list[i]:], AR_verify_tokens[i][start_pos_list[i]:-1])):
                if token not in AR_verify_token:
                    bifurcation = j
                    break
            next_output_tokens[i] = prev_output_tokens[i][:start_pos_list[i] + bifurcation] + \
                                    [AR_verify_tokens[i][start_pos_list[i] + bifurcation][0]] + \
                                    [tgt_dict.unk()] * block_size

            find_eos = False
            for j, o in enumerate(next_output_tokens[i][start_pos_list[i]:start_pos_list[i] + bifurcation + 1]):
                if o == tgt_dict.eos() or start_pos_list[i] + j == max_len:
                    next_output_tokens[i] = next_output_tokens[i][:start_pos_list[i] + j]
                    start_pos_list[i] = -1
                    find_eos = True
                    break
            if not find_eos:
                start_pos_list[i] = start_pos_list[i] + bifurcation + 1

    return next_output_tokens, start_pos_list


if __name__ == '__main__':
    parser = options.get_generation_parser()
    parser.add_argument('--input-path', type=str, required=True,
                        help='path to eval file')
    parser.add_argument('--output-path', type=str, default=None,
                        help='path to output file')
    parser.add_argument('--AR-path', type=str, default=None,
                        help='path to AR model')
    parser.add_argument('--strategy', type=str, default='fairseq',
                        help='decoding strategy, choose from: fairseq, AR, gad')
    parser.add_argument('--batch', type=int, default=None,
                        help='batch size')
    parser.add_argument('--block-size', type=int, default=5,
                        help='block size')
    parser.add_argument('--beta', type=int, default=1,
                        help='top-beta hyperparameter')
    parser.add_argument('--tau', type=float, default=0,
                        help='tolerance hyperparameter')
    cmd_args = options.parse_args_and_arch(parser)
    cmd_args.input_path = os.path.expanduser(cmd_args.input_path)
    cmd_args.output_path = os.path.expanduser(cmd_args.output_path)

    cfg = convert_namespace_to_omegaconf(cmd_args)

    task = tasks.setup_task(cfg.task)

    # NAR drafter
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, _model_args, _model_task = load_model_ensemble_and_task(filenames=[cfg.common_eval.path], task=task)

    if cmd_args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    model = models[0].to(device).eval()
    if cfg.common.fp16:
        logging.info("NAR fp16 enabled!")
        model.half()

    # AR verifier
    AR_model = None
    AR_models = None
    _AR_model_task = None
    if cmd_args.AR_path is not None:
        AR_models, _AR_model_args, _AR_model_task = load_model_ensemble_and_task(filenames=[cmd_args.AR_path],
                                                                                 arg_overrides={'data': cfg.task.data})
        if cfg.common.fp16:
            logging.info("AR fp16 enabled!")
            for AR_model in AR_models:
                AR_model.half()
        AR_model = AR_models[0].to(device).eval()
        logging.info("AR model loaded!")

    with open(cmd_args.input_path, 'r') as f:
        bpe_sents = [l.strip() for l in f.readlines()]

    if cmd_args.strategy == 'AR':
        logger.info("Decoding Strategy: Simplified AR")
        remove_bpe_results, delta = baseline_generate(bpe_sents, AR_model, _AR_model_task, cmd_args.batch, device)
        logger.info(f'Simplified AR generate: {delta}')
    elif cmd_args.strategy == 'gad':
        logger.info("Decoding Strategy: GAD")
        remove_bpe_results, delta = gad_generate(bpe_sents, model, AR_model, task, cmd_args.block_size, cmd_args.batch,
                                                 device, beta=cmd_args.beta, tau=cmd_args.tau)
        logger.info(f'GAD generate: {delta}')
    else:
        logger.info("Decoding Strategy: fairseq")
        remove_bpe_results, delta = fairseq_generate(bpe_sents, cfg, AR_models, _AR_model_task, cmd_args.batch, device)
        logger.info(f'Fairseq generate batch {cmd_args.batch}, beam {cfg.generation.beam}: {delta}')

    if cmd_args.output_path is not None:
        write_result(remove_bpe_results, cmd_args.output_path)
