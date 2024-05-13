import torch
import torch.nn as nn
from argparse import Namespace
from torchscale.component.legacy_multihead_attention import MultiheadAttention

def build_connector(args, input_dim, output_dim):
    connector = XConnector(input_dim, output_dim, args)
    return connector

class XConnector(nn.Module):
    """Connector model of GPT and MLM."""

    def __init__(self, input_dim, output_dim, args, ):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.latent_query = torch.nn.Parameter(torch.randn(args.latent_query_num, output_dim))

        ts_args = Namespace(**{'scale_length': 0, 'multiway': False, 'flash_attention': False})
        self.x_attn = MultiheadAttention(ts_args, output_dim, args.decoder_attention_heads,
                                         dropout=args.attention_dropout, encoder_decoder_attention=True)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = x.view(-1, kwargs['src_len'], x.size(-1)).transpose(0, 1)
        bsz = x.size(1)
        latent_query = self.latent_query.unsqueeze(1).expand(-1, bsz, -1)
        x, _ = self.x_attn(latent_query, torch.cat([x, latent_query]), torch.cat([x, latent_query]))
        return x.transpose(0, 1).contiguous().view(-1, x.size(-1))
