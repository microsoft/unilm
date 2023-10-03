import torch
import torch.nn as nn
from fairseq.modules import MultiheadAttention
from fairseq import utils


def build_connector(args, input_dim, output_dim):
    connector_name = args.text_connector if hasattr(args, "text_connector") else args.connector
    if connector_name == "none":
        connector = None
    elif connector_name == "complex":
        connector = ComplexConnector(input_dim, 
                                    output_dim,
                                    args.activation_fn)
    elif connector_name == "simple":
        connector = SimpleConnector(input, 
                                    output_dim)
    elif connector_name == "xconnector":
        connector = XConnector(input_dim, output_dim, args)
    else:
        raise ValueError("Invalid text connector type: {}".format(args.connector))
    return connector


class SimpleConnector(nn.Module):
    """Connector model of GPT and MLM."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, features, **kwargs):
        
        x = self.dense(features)
        return x


class ComplexConnector(nn.Module):
    """Connector model of GPT and MLM."""

    def __init__(self, input_dim, output_dim, activation_fn):
        super().__init__()
        self.dense = nn.Linear(input_dim, input_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.predict = nn.Linear(input_dim, output_dim)

    def forward(self, features, **kwargs):

        x = self.dense(features)
        x = self.activation_fn(x)

        x = self.predict(x)
        return x


class XConnector(nn.Module):
    """Connector model of GPT and MLM."""

    def __init__(self, input_dim, output_dim, args,):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.latent_query = torch.nn.Parameter(torch.randn(args.latent_query_num, output_dim)) 
        
        self.x_attn = MultiheadAttention(
            output_dim,
            args.decoder_attention_heads,
            kdim=output_dim,
            vdim=output_dim,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )

    def forward(self, features, **kwargs):

        x = self.dense(features)
        # x = attention_i(q=latent_query, kv=concat([x, latent_query]))
        # shape of x is [batch_size * seq_len, output_dim] -> [seq_len, batch_size, output_dim]
        x = x.view(-1, kwargs['src_len'], x.size(-1)).transpose(0, 1)
        bsz = x.size(1)
        latent_query = self.latent_query.unsqueeze(1).expand(-1, bsz, -1)
        x, _ = self.x_attn(latent_query, torch.cat([x, latent_query]), torch.cat([x, latent_query]))        
        return x.transpose(0, 1).contiguous().view(-1, x.size(-1)) # [batch_size * seq_len, output_dim]

