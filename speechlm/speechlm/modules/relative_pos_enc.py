# --------------------------------------------------------
# Pre-Training Transformer Decoder for End-to-End ASR Model with Unpaired Speech Data (https://arxiv.org/abs/2203.17113)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/Speech2C
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

import torch

class RelativePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, maxlen=1000, embed_v=False):
        super(RelativePositionalEncoding, self).__init__()

        self.d_model = d_model
        self.maxlen = maxlen
        self.pe_k = torch.nn.Embedding(2*maxlen, d_model) 
        if embed_v:
            self.pe_v = torch.nn.Embedding(2*maxlen, d_model)
        self.embed_v = embed_v


    def forward(self, pos_seq, incremental_state=None):
        pos_seq[pos_seq < -self.maxlen] = -self.maxlen
        pos_seq[pos_seq >= self.maxlen] = self.maxlen - 1
        pos_seq = pos_seq + self.maxlen
        
        if incremental_state is not None:
            pos_seq = pos_seq[-1:]

        if self.embed_v:
            return self.pe_k(pos_seq), self.pe_v(pos_seq)
        else:
            return self.pe_k(pos_seq), None
