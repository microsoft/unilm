from omegaconf import DictConfig

from general_util.logger import get_child_logger

from lightseq.training.ops.pytorch.transformer_encoder_layer import (
    LSTransformerEncoderLayer,
)

logger = get_child_logger("LightSeqUtils")


class LSHFTransformerEncoderLayer(LSTransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(LSHFTransformerEncoderLayer, self).__init__(*args, **kwargs)

    def forward(self, hidden_states, encoder_padding_mask, *args, **kwargs):
        ls_encoder_padding_mask = encoder_padding_mask / -10000.0
        ls_encoder_padding_mask = ls_encoder_padding_mask.squeeze()
        output = super().forward(hidden_states, ls_encoder_padding_mask)
        return output, None, None, None


def gen_bert_config(cfg: DictConfig, config):
    bert_config = LSTransformerEncoderLayer.get_config(
        max_batch_tokens=4096,
        max_seq_len=config.max_position_embeddings,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        nhead=config.num_attention_heads,
        attn_prob_dropout_ratio=config.attention_probs_dropout_prob,
        activation_dropout_ratio=config.hidden_dropout_prob,
        hidden_dropout_ratio=config.hidden_dropout_prob,
        pre_layer_norm=False,
        fp16=cfg.fp16,
        local_rank=cfg.local_rank,
        activation_fn="gelu",
    )
    return bert_config


def get_hf_bert_enc_layer_params(layer):
    init_ws = []
    init_bs = []

    init_ws.append(layer.attention.self.query.weight.detach().clone())
    init_bs.append(layer.attention.self.query.bias.detach().clone())
    init_ws.append(layer.attention.self.key.weight.detach().clone())
    init_bs.append(layer.attention.self.key.bias.detach().clone())
    init_ws.append(layer.attention.self.value.weight.detach().clone())
    init_bs.append(layer.attention.self.value.bias.detach().clone())
    init_ws.append(layer.attention.output.dense.weight.detach().clone())
    init_bs.append(layer.attention.output.dense.bias.detach().clone())
    init_ws.append(layer.attention.output.LayerNorm.weight.detach().clone())
    init_bs.append(layer.attention.output.LayerNorm.bias.detach().clone())

    init_ws.append(layer.intermediate.dense.weight.detach().clone())
    init_bs.append(layer.intermediate.dense.bias.detach().clone())
    init_ws.append(layer.output.dense.weight.detach().clone())
    init_bs.append(layer.output.dense.bias.detach().clone())
    init_ws.append(layer.output.LayerNorm.weight.detach().clone())
    init_bs.append(layer.output.LayerNorm.bias.detach().clone())

    return init_ws, init_bs


def inject_ls_enc_layer(model, cfg, config):
    for i in range(config.num_hidden_layers):
        bert_config = gen_bert_config(cfg, config)
        init_ws, init_bs = get_hf_bert_enc_layer_params(model.bert.encoder.layer[i])
        model.bert.encoder.layer[i] = LSHFTransformerEncoderLayer(
            bert_config, init_ws, init_bs
        ).cuda()


def inject_ls_roberta_enc_layer(model, cfg, config):
    for i in range(config.num_hidden_layers):
        bert_config = gen_bert_config(cfg, config)
        init_ws, init_bs = get_hf_bert_enc_layer_params(model.roberta.encoder.layer[i])
        model.roberta.encoder.layer[i] = LSHFTransformerEncoderLayer(
            bert_config, init_ws, init_bs
        )

