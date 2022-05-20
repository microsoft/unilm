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
def baseline_generate(data_lines, model, task, device, max_len=200):
    # simplified AR greedy decoding
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    data_size = len(data_lines)
    all_results = []
    logger.info(f'Baseline generate')
    start = time.perf_counter()
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
            pred_token = baseline_forward_decoder(model,
                                                  cur_input_tokens,
                                                  encoder_out,
                                                  incremental_state).item()
            if pred_token == tgt_dict.eos():
                break
            else:
                tokens.append(pred_token)
        all_results.append(tgt_dict.string(tokens[1:]))
    delta = time.perf_counter() - start
    remove_bpe_results = [line.replace('@@ ', '') for line in all_results]
    return remove_bpe_results, delta


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
def forward_decoder(model,
                    input_tokens,
                    encoder_out: Dict[str, List[Tensor]],
                    incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
                    parallel_forward_start_pos=None,
                    temperature: float = 1.0,
                    beta: int = 1,
                    tau: float = 0.0):
    decoder_out = model.decoder.forward(input_tokens,
                                        encoder_out=encoder_out,
                                        incremental_state=incremental_state,
                                        parallel_forward_start_pos=parallel_forward_start_pos)
    decoder_out_tuple = (decoder_out[0].div_(temperature), decoder_out[1])
    topk_scores, indexes = torch.topk(decoder_out_tuple[0], beta, dim=-1)
    topk_scores = topk_scores[0].tolist()
    indexes = indexes[0].tolist()
    for i in range(len(topk_scores)):
        for j, s in enumerate(topk_scores[i]):
            if topk_scores[i][0] - s > tau:
                indexes[i][j] = -1
    return indexes


def gad_generate(data_lines, model, AR_model, task, block_size, device, beta=1, tau=0, max_len=200):
    # Generalized Aggressive Decoding
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    encoder_state_ids = []
    for i in range(len(AR_model.decoder.layers)):
        encoder_state_ids.append(AR_model.decoder.layers[i].encoder_attn._incremental_state_id)
    data_size = len(data_lines)
    all_results = []
    logger.info(f'GAD generate')
    pass_tokens = [0] * max_len
    sent_nums = [0] * max_len
    start = time.perf_counter()
    for start_idx in tqdm(range(0, data_size)):
        bpe_line = data_lines[start_idx]
        src_tokens = src_dict.encode_line(bpe_line, add_if_not_exist=False).long()
        net_input = {'src_tokens': src_tokens.unsqueeze(0).to(device),
                     'src_lengths': torch.LongTensor([src_tokens.numel()]).to(device)}
        AR_encoder_out = AR_model.encoder.forward_torchscript(net_input)
        encoder_out = model.encoder.forward_torchscript(net_input)
        incremental_state = torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]],
                                               torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}))
        prev_output_tokens = [tgt_dict.unk()] * block_size
        start_pos = 0
        for step in range(0, max_len):
            start_pos, prev_output_tokens, pass_token = gad_forward(incremental_state, encoder_state_ids,
                                                                    start_pos, block_size, tgt_dict,
                                                                    prev_output_tokens,
                                                                    encoder_out, AR_encoder_out, model,
                                                                    AR_model, beta, tau)
            pass_tokens[step] += pass_token
            sent_nums[step] += 1
            if start_pos == -1:
                break
        all_results.append(tgt_dict.string(prev_output_tokens))
    total_pass_tokens = 0
    total_sent_nums = 0
    for step in range(max_len):
        if sent_nums[step] > 0:
            total_pass_tokens += pass_tokens[step]
            total_sent_nums += sent_nums[step]
    print("Avg accepted tokens:", total_pass_tokens / total_sent_nums)
    total_iter = 0
    for step in range(max_len):
        if sent_nums[step - 1] > 0:
            if step == 0:
                last_num = data_size
            else:
                last_num = sent_nums[step - 1]
            if (last_num - sent_nums[step]) > 0:
                total_iter += (last_num - sent_nums[step]) * (step)
    print("Avg decoding iteration:", total_iter / data_size)
    delta = time.perf_counter() - start
    remove_bpe_results = [line.replace('@@ ', '') for line in all_results]
    return remove_bpe_results, delta


def gad_forward(incremental_state, encoder_state_ids, start_pos, block_size, tgt_dict, prev_output_tokens,
                encoder_out, AR_encoder_out, model, AR_model, beta, tau, max_len=200):
    output_tokens = torch.tensor([prev_output_tokens]).to(device)
    _scores, _tokens = model.decoder(
        normalize=False,
        prev_output_tokens=output_tokens,
        encoder_out=encoder_out,
    ).max(-1)

    prev_output_tokens[start_pos:start_pos + block_size] = _tokens[0].tolist()[start_pos:start_pos + block_size]

    cut_incremental_state(incremental_state, keep_len=start_pos, encoder_state_ids=encoder_state_ids)
    cur_span_input_tokens = torch.tensor([[tgt_dict.eos()] + prev_output_tokens]).to(device)
    AR_topk_tokens = forward_decoder(AR_model,
                                     cur_span_input_tokens,
                                     AR_encoder_out,
                                     incremental_state,
                                     parallel_forward_start_pos=start_pos,
                                     beta=beta,
                                     tau=tau)

    bifurcation = block_size
    for i, (token, AR_topk_token) in enumerate(zip(prev_output_tokens[start_pos:], AR_topk_tokens[:-1][:])):
        if token not in AR_topk_token:
            bifurcation = i
            break

    next_output_tokens = prev_output_tokens[:start_pos + bifurcation] + [AR_topk_tokens[bifurcation][0]] + [
        tgt_dict.unk()] * block_size

    pass_token = 0
    find_eos = False
    for i, o in enumerate(next_output_tokens[start_pos:start_pos + bifurcation + 1]):
        if o == tgt_dict.eos() or i + start_pos == max_len:
            next_output_tokens = next_output_tokens[0:start_pos + i]
            start_pos = -1
            pass_token = i
            find_eos = True
            break

    if not find_eos:
        start_pos = start_pos + bifurcation + 1
        pass_token = bifurcation + 1

    return start_pos, next_output_tokens, pass_token


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

    # AR verifier
    AR_model = None
    AR_models = None
    _AR_model_task = None
    if cmd_args.AR_path is not None:
        AR_models, _AR_model_args, _AR_model_task = load_model_ensemble_and_task(filenames=[cmd_args.AR_path],
                                                                                 arg_overrides={'data': cfg.task.data})
        AR_model = AR_models[0].to(device).eval()
        logging.info("AR model loaded!")

    with open(cmd_args.input_path, 'r') as f:
        bpe_sents = [l.strip() for l in f.readlines()]

    if cmd_args.strategy == 'AR':
        logger.info("Decoding Strategy: Simplified AR")
        remove_bpe_results, delta = baseline_generate(bpe_sents, AR_model, _AR_model_task, device)
        logger.info(f'Simplified AR generate: {delta}')
    elif cmd_args.strategy == 'gad':
        logger.info("Decoding Strategy: GAD")
        remove_bpe_results, delta = gad_generate(bpe_sents, model, AR_model, task, cmd_args.block_size, device,
                                                 beta=cmd_args.beta, tau=cmd_args.tau)
        logger.info(f'GAD generate: {delta}')
    else:
        logger.info("Decoding Strategy: fairseq")
        remove_bpe_results, delta = fairseq_generate(bpe_sents, cfg, AR_models, _AR_model_task, cmd_args.batch, device)
        logger.info(f'Fairseq generate batch {cmd_args.batch}, beam {cfg.generation.beam}: {delta}')

    if cmd_args.output_path is not None:
        write_result(remove_bpe_results, cmd_args.output_path)
