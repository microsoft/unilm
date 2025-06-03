import time
import torch
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group
from typing import Optional
import datetime
import lm_eval
from lm_eval.models.huggingface import HFLM as eval_wrapper
from lm_eval.tasks import get_task_dict, TaskManager
from lm_eval.evaluator import evaluate
from eval_math import evaluate as evaluate_math
import os, json
from data.tokenizer import Tokenizer
from config import parse_eval_args
from arch.model import ModelArgs, Model, create_kv_cache
import re
import torch.nn.functional as F
from safetensors.torch import load_file


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token.squeeze(-1)

class EvalWrapper(eval_wrapper):
    def __init__(
        self,
        model,
        tokenizer,
        batch_size,
        max_seq_length: Optional[int]=None,
    ):
        super().__init__(pretrained="gpt2")
        self._model = model
        self._tokenizer = tokenizer
        self._device = torch.device('cuda')
        self._max_seq_length = 2048 if max_seq_length is None else max_seq_length
        self._batch_size = batch_size
        self._rank = 0
        self._world_size = 1

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id
    
    @property
    def eos_token_id(self):
        return self._tokenizer.eos_id
    
    @property
    def pad_token_id(self):
        return self._tokenizer.pad_id

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return 1024

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device
    
    @property
    def model(self):
        return self._model

    def tok_encode(self, string: str, **kwargs):
        encoded = self._tokenizer.encode(string, bos=True, eos=False)
        return encoded
    
    def tok_decode(self, tokens, **kwargs):
        if type(tokens) == int:
            tokens = [tokens]
        decoded = self._tokenizer.decode(tokens)
        return decoded
    
    def tok_batch_encode(self, strings, left_truncate_len=None, **kwargs):
        tokens = [self._tokenizer.encode(string, bos=True, eos=False) for string in strings]
        if left_truncate_len is not None:
            tokens = [t[-left_truncate_len:] for t in tokens]
        max_len = max(len(t) for t in tokens)
        tokens = [t + [self.pad_token_id] * (max_len - len(t)) for t in tokens]
        tensor = torch.tensor(tokens).long()
        return tensor, tensor # return a dummy tensor for the attention mask

    def _model_call(self, inps):
        logits = self.model(inps)
        return logits

    def _model_generate(self, context, max_length, kv_cache=None, **generation_kwargs):
        bsz = context.size(0)
        tokens = context.tolist()
        PREFILL_CHUNK_SIZE = 65536 // bsz
        # remove padding
        tokens = [t[:t.index(self.pad_token_id)] if self.pad_token_id in t else t for t in tokens]
        seqlens = torch.tensor([len(t) for t in tokens], device=self.device)
        max_seqlen = seqlens.max().item()
        generation_length = max_length - max_seqlen

        eos_reached = torch.tensor([False] * bsz, device=self.device)
        kv_cache = create_kv_cache(self.model.args, bsz) if kv_cache is None else kv_cache
        for layer_kv_cache in kv_cache:
            layer_kv_cache[2].clear_centeroids()
        output = torch.zeros(bsz, generation_length, dtype=torch.long, device=self.device).fill_(self.pad_token_id)
        for cur_pos in range(generation_length):
            if cur_pos == 0:
                last_logits = torch.zeros(bsz, self.model.args.vocab_size, device=self.device, dtype=torch.float16)
                for pre_start_pos in range(0, max_seqlen, PREFILL_CHUNK_SIZE):
                    pre_end_pos = min(pre_start_pos + PREFILL_CHUNK_SIZE, max_seqlen)
                    chunk_start_pos = torch.tensor([min(seqlens[b], pre_start_pos) for b in range(bsz)], device=self.device)
                    chunk_end_pos = torch.tensor([min(seqlens[b], pre_end_pos) for b in range(bsz)], device=self.device)
                    chunk_tokens = torch.cat([torch.tensor(tokens[b][chunk_start_pos[b]:chunk_end_pos[b]], device=self.device) for b in range(bsz)], dim=0)
                    cu_seqlens = torch.cat([torch.tensor([0], device=self.device), (chunk_end_pos - chunk_start_pos).cumsum(dim=0)], dim=0)
                    logits = self.model(chunk_tokens[None, :], start_pos=chunk_start_pos, cu_seqlens=cu_seqlens, kv_cache=kv_cache, last_hidden_only=True)
                    is_last = (chunk_end_pos == seqlens) & (chunk_start_pos < chunk_end_pos)
                    last_logits[is_last] = self.model.output(logits[0, cu_seqlens[1:][is_last] - 1])
                # prefill_tokens = torch.cat([torch.tensor(tokens[b], device=self.device) for b in range(bsz)], dim=0)
                # cu_seqlens = torch.cat([torch.tensor([0], device=self.device), seqlens.cumsum(dim=0)], dim=0)
                # logits = self.model(prefill_tokens[None, :], start_pos=torch.zeros(bsz, device=self.device, dtype=torch.long), cu_seqlens=cu_seqlens, kv_cache=kv_cache, last_hidden_only=True)
            else:
                next_input = torch.where(eos_reached[:, None], 0, output[:, cur_pos - 1:cur_pos])
                logits = self.model(next_input, start_pos=seqlens + cur_pos - 1, cu_seqlens=torch.arange(bsz + 1, device=self.device), kv_cache=kv_cache, last_hidden_only=True)
                last_logits = self.model.output(logits[:, -1, :])
            if self.model.args.temperature > 0:
                probs = torch.softmax(last_logits / self.model.args.temperature, dim=-1)
                next_tokens = sample_top_p(probs, self.model.args.top_p).reshape(-1)
            else:
                next_tokens = torch.argmax(last_logits, dim=-1).reshape(-1)
            output[:, cur_pos] = torch.where(eos_reached, output[:, cur_pos], next_tokens)
            eos_reached |= (next_tokens == self.eos_token_id)
            if eos_reached.all():
                break

            if cur_pos > 0 and self.model.args.resa_rec_freq > 0 and cur_pos % self.model.args.resa_rec_freq == 0:
                start_pos = cur_pos - self.model.args.resa_rec_freq
                reprefill_tokens = output[:, start_pos:start_pos + self.model.args.resa_rec_freq]
                reprefill_tokens = torch.where(reprefill_tokens == self.pad_token_id, 0, reprefill_tokens)
                logits = self.model(reprefill_tokens, start_pos=seqlens + start_pos, cu_seqlens=torch.arange(bsz + 1, device=self.device) * self.model.args.resa_rec_freq, kv_cache=kv_cache, last_hidden_only=True)
                for layer_kv_cache in kv_cache:
                    # HOTFIX: clear centeroids for each layer, online modification in the future
                    layer_kv_cache[2].clear_centeroids()

        final_output = torch.full((bsz, max_length), fill_value=self.pad_token_id, dtype=torch.long, device=self.device)
        final_output[:, :max_seqlen] = context
        for b in range(bsz):
            final_output[b, seqlens[b]:seqlens[b] + generation_length] = output[b]
        return final_output

def _adjust_config(task_dict):
    adjusted_task_dict = {}
    for task_name, task_obj in task_dict.items():
        if isinstance(task_obj, dict):
            adjusted_task_dict = {
                **adjusted_task_dict,
                **{task_name: _adjust_config(task_obj)},
            }
        else:
            if 'mmlu' in task_name or 'gsm' in task_name:
                task_obj.set_config(key="num_fewshot", value=5)
            elif task_obj.get_config("num_fewshot") is None:
                task_obj.set_config(key="num_fewshot", value=0)
            task_obj.set_fewshot_seed(seed=1234)
            adjusted_task_dict[task_name] = task_obj

    return adjusted_task_dict

@torch.no_grad()
def eval_end_task(
    model,
    tokenizer,
    tasks,
    limit,
    batch_size,
    max_seq_length,
):
    model_eval_wrapper = EvalWrapper(
        model,
        tokenizer,
        batch_size,
        max_seq_length,
    )

    task_dict = get_task_dict(tasks.split(','), task_manager=TaskManager(verbosity='WARNING'))
    task_dict = _adjust_config(task_dict)

    eval_results = evaluate(
        model_eval_wrapper,
        task_dict,
        limit=limit,
        verbosity='WARNING',
    )
    return eval_results

@torch.no_grad()
def eval_downstream_task(
    args,
    model,
    tokenizer,
    downstream_task,
    limit,
    batch_size,
    max_seq_length,
):
    model_eval_wrapper = EvalWrapper(
        model,
        tokenizer,
        batch_size,
        max_seq_length,
    )

    if downstream_task == "math":
        evaluate_func = evaluate_math
    else:
        raise ValueError(f"Unknown downstream task: {downstream_task}")
    
    evaluate_func(
        args,
        model_eval_wrapper,
        limit=limit,
    )

def load_qwen2_model(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if "lm_head" in k:
            new_state_dict[k.replace("lm_head", "output")] = v
        elif "embed_tokens" in k:
            new_state_dict[k.replace("model.embed_tokens", "tok_embeddings")] = v
        else:
            new_state_dict[k.replace("model.", "")] = v
    return new_state_dict

def load_model(args):
    tokenizer = Tokenizer(args.checkpoint_dir)
    config = json.load(open(os.path.join(args.checkpoint_dir, "config.json")))
    params = {
        "dim": config["hidden_size"],
        "hidden_dim": config["intermediate_size"],
        "n_layers": config["num_hidden_layers"],
        "n_heads": config["num_attention_heads"],
        "n_kv_heads": config["num_key_value_heads"],
        "rope_theta": config["rope_theta"],
        "norm_eps": config["rms_norm_eps"],
        "vocab_size": config["vocab_size"],
        "max_batch_size": args.batch_size,
        "max_seq_len": config["max_position_embeddings"],
        "tie_word_embeddings": config["tie_word_embeddings"],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "resa_rec_freq": args.resa_rec_freq,
        "resa_block_size": args.resa_block_size,
        "resa_local_block_num": args.resa_local_block_num,
        "resa_min_block_num": args.resa_min_block_num,
        "resa_sparse_ratio": args.resa_sparse_ratio,
    }
    model_args = ModelArgs(**params)
    if "model.safetensors.index.json" in os.listdir(args.checkpoint_dir):
        safetensor_file = filter(lambda x: x.endswith(
            "safetensors"), os.listdir(args.checkpoint_dir))
        state_dict = {}
        for file in safetensor_file:
            state_dict.update(load_qwen2_model(load_file(os.path.join(
                args.checkpoint_dir, file))))
    else:
        state_dict = load_qwen2_model(load_file(os.path.join(
            args.checkpoint_dir, "model.safetensors")))
        
    for i in range(model_args.n_layers):
        state_dict[f"layers.{i}.self_attn.qkv_proj.weight"] = torch.cat([state_dict[f"layers.{i}.self_attn.q_proj.weight"], state_dict[f"layers.{i}.self_attn.k_proj.weight"], state_dict[f"layers.{i}.self_attn.v_proj.weight"]], dim=0)
        state_dict[f"layers.{i}.self_attn.qkv_proj.bias"] = torch.cat([state_dict[f"layers.{i}.self_attn.q_proj.bias"], state_dict[f"layers.{i}.self_attn.k_proj.bias"], state_dict[f"layers.{i}.self_attn.v_proj.bias"]], dim=0)
        del state_dict[f"layers.{i}.self_attn.q_proj.weight"], state_dict[f"layers.{i}.self_attn.k_proj.weight"], state_dict[f"layers.{i}.self_attn.v_proj.weight"]
        del state_dict[f"layers.{i}.self_attn.q_proj.bias"], state_dict[f"layers.{i}.self_attn.k_proj.bias"], state_dict[f"layers.{i}.self_attn.v_proj.bias"]
        
    model = Model(model_args)
    model = model.cuda().to(dtype=torch.float16)
    model.eval()
    model.load_state_dict(state_dict, strict=True)
    return model, tokenizer

def evaluate_one_checkpoint(args):
    model, tokenizer = load_model(args)

    if args.tasks is not None:
        results = eval_end_task(
            model,
            tokenizer,
            args.tasks,
            args.limit,
            args.batch_size,
            model.args.max_seq_len,
        )
        
        for task, res in results["results"].items():
            if task in args.tasks.split(','):
                print(f"{task}: {res}")

        return results
    elif args.downstream_task is not None:
        eval_downstream_task(
            args,
            model,
            tokenizer,
            args.downstream_task,
            args.limit,
            args.batch_size,
            model.args.max_seq_len,
        )
        return None
    else:
        raise NotImplementedError("No evaluation task specified")

if __name__ == '__main__':
    init_process_group(backend='gloo', timeout=datetime.timedelta(hours=2))
    dp_rank = int(os.environ['RANK'])
    dp_local_rank = int(os.environ['LOCAL_RANK'])
    dp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{dp_local_rank}'
    torch.cuda.set_device(device)

    args = parse_eval_args()
    results = evaluate_one_checkpoint(args)
