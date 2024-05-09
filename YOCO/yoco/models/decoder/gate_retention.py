import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.model_parallel.megatron.mpu import (
    ColumnParallelLinear,
    RowParallelLinear,
)

from .rms_norm import RMSNorm

from .kernel.gate_recurrent import chunk_gate_retention, recurrent_gate_retention
from .kernel.rotary import apply_rotary_emb
from .kernel.swiglu import swiglu

from .model_parallel_init import qkvg_init_method, out_init_method

class GateRetention(nn.Module):

    def __init__(
        self,
        args,
        gate_logit_normalizer: int = 16,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = args.dim
        self.num_heads = args.n_self_heads // args.model_parallel_size
        self.head_dim = args.dim // args.n_self_heads

        self.q_proj = ColumnParallelLinear(args.dim, args.dim, bias=False, gather_output=False, init_method=qkvg_init_method)
        self.k_proj = ColumnParallelLinear(args.dim, args.dim, bias=False, gather_output=False, init_method=qkvg_init_method)
        self.v_proj = ColumnParallelLinear(args.dim, args.dim, bias=False, gather_output=False, init_method=qkvg_init_method)
        self.g_proj = ColumnParallelLinear(args.dim, args.dim, bias=False, gather_output=False, init_method=qkvg_init_method)
        self.gt_proj = ColumnParallelLinear(args.dim, args.n_self_heads, bias=False, gather_output=False, init_method=qkvg_init_method)
        
        self.out_proj = RowParallelLinear(args.dim, args.dim, bias=False, input_is_parallel=True, init_method=out_init_method)

        self.subln = RMSNorm(self.head_dim, elementwise_affine=False, eps=args.norm_eps)

        self.gate_logit_normalizer = gate_logit_normalizer

    def forward(
        self,
        x,
        rel_pos,
        incremental_state=None,
        is_prefilling=False,
    ):
        bsz, tgt_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)
        gt = self.gt_proj(x)

        qr = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
        kr = k.view(bsz, tgt_len, self.num_heads, self.head_dim)
        v = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        gt = gt.view(bsz, tgt_len, self.num_heads).transpose(1, 2)

        qr = apply_rotary_emb(qr, *rel_pos, interleaved=True).transpose(1, 2)
        kr = apply_rotary_emb(kr, *rel_pos, interleaved=True).transpose(1, 2)
        gt = (F.logsigmoid(gt) / self.gate_logit_normalizer)

        if incremental_state is not None and not is_prefilling:
            o = recurrent_gate_retention(qr, kr, v, gt, incremental_state)
        else:
            if incremental_state is not None:
                index_mask = incremental_state["index_mask"]
                gt_sum = gt.float().masked_fill(index_mask, 0).sum(dim=-1, keepdim=True)
                gt_mask = (gt_sum - gt.float().cumsum(dim=-1)).exp().masked_fill(index_mask, 0)
                next_hidden_state = (kr.transpose(-1, -2) * (self.head_dim ** -0.5)) @ (v * gt_mask.to(v.dtype).unsqueeze(-1))
                if "last_hidden_state" in incremental_state:
                    last_hidden_state = incremental_state["last_hidden_state"]
                    next_hidden_state += last_hidden_state * gt_sum.exp().unsqueeze(-1).to(v.dtype) if last_hidden_state is not None else 0
                else:
                    last_hidden_state = None
                incremental_state["last_hidden_state"] = next_hidden_state
                o = chunk_gate_retention(qr, kr, v, gt, chunk_size=256, last_hidden_state=last_hidden_state)
            else:
                o = chunk_gate_retention(qr, kr, v, gt, chunk_size=256)
                
        o = self.subln(o).transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * self.head_dim)
        o = swiglu(g, o)
        o = self.out_proj(o)
        return o
