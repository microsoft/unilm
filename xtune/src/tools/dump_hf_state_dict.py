import logging
import torch

from collections import OrderedDict

from transformers.modeling_bert import (BertConfig, BertEncoder,
                                        BertIntermediate, BertLayer,
                                        BertModel, BertOutput,
                                        BertSelfAttention,
                                        BertSelfOutput)
from transformers.modeling_roberta import (RobertaEmbeddings,
                                           RobertaForMaskedLM,
                                           RobertaForSequenceClassification,
                                           RobertaModel)

logger = logging.getLogger(__name__)


# NOTE transformers should be 2.5.1

def convert_cxlm_to_transformers(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]

    config = BertConfig(
        # vocab_size_or_config_json_file=250002,
        vocab_size=250002,
        hidden_size=args.encoder_embed_dim,
        num_hidden_layers=args.encoder_layers,
        num_attention_heads=args.encoder_attention_heads,
        intermediate_size=args.encoder_ffn_embed_dim,
        max_position_embeddings=args.max_positions + 2,
        type_vocab_size=1,
        layer_norm_eps=1e-5,  # PyTorch default used in fairseq
    )

    print("Our BERT config:", config)

    stat_dict = ckpt["model"]
    new_stat_dict = {}

    model = RobertaForMaskedLM(config)
    model.eval()

    sent_enc = "decoder.sentence_encoder"
    new_stat_dict["roberta.embeddings.word_embeddings.weight"] = stat_dict[sent_enc + ".embed_tokens.weight"]
    new_stat_dict["roberta.embeddings.position_embeddings.weight"] = stat_dict[sent_enc + ".embed_positions.weight"]

    new_stat_dict["roberta.embeddings.token_type_embeddings.weight"] = torch.zeros_like(
        model.roberta.embeddings.token_type_embeddings.weight)

    new_stat_dict["roberta.embeddings.LayerNorm.weight"] = stat_dict[sent_enc + ".emb_layer_norm.weight"]
    new_stat_dict["roberta.embeddings.LayerNorm.bias"] = stat_dict[sent_enc + ".emb_layer_norm.bias"]

    for i in range(config.num_hidden_layers):
        # Encoder: start of layer
        # layer: BertLayer = model.roberta.encoder.layer[i]
        layer = "roberta.encoder.layer.%d" % i
        roberta_layer = sent_enc + (".layers.%d" % i)

        ### self attention
        # self_attn: BertSelfAttention = layer.attention.self
        self_attn = layer + ".attention.self"
        assert (
                stat_dict[roberta_layer + ".self_attn.k_proj.weight"].data.shape == \
                stat_dict[roberta_layer + ".self_attn.q_proj.weight"].data.shape == \
                stat_dict[roberta_layer + ".self_attn.v_proj.weight"].data.shape == \
                torch.Size((config.hidden_size, config.hidden_size))
        )

        new_stat_dict[self_attn + ".query.weight"] = stat_dict[roberta_layer + ".self_attn.q_proj.weight"]
        new_stat_dict[self_attn + ".query.bias"] = stat_dict[roberta_layer + ".self_attn.q_proj.bias"]
        new_stat_dict[self_attn + ".key.weight"] = stat_dict[roberta_layer + ".self_attn.k_proj.weight"]
        new_stat_dict[self_attn + ".key.bias"] = stat_dict[roberta_layer + ".self_attn.k_proj.bias"]
        new_stat_dict[self_attn + ".value.weight"] = stat_dict[roberta_layer + ".self_attn.v_proj.weight"]
        new_stat_dict[self_attn + ".value.bias"] = stat_dict[roberta_layer + ".self_attn.v_proj.bias"]

        ### self-attention output
        # self_output: BertSelfOutput = layer.attention.output
        self_output = layer + ".attention.output"
        assert (
                model.roberta.encoder.layer[i].attention.output.dense.weight.shape == stat_dict[
            roberta_layer + ".self_attn.out_proj.weight"].shape
        )
        new_stat_dict[self_output + ".dense.weight"] = stat_dict[roberta_layer + ".self_attn.out_proj.weight"]
        new_stat_dict[self_output + ".dense.bias"] = stat_dict[roberta_layer + ".self_attn.out_proj.bias"]
        new_stat_dict[self_output + ".LayerNorm.weight"] = stat_dict[roberta_layer + ".self_attn_layer_norm.weight"]
        new_stat_dict[self_output + ".LayerNorm.bias"] = stat_dict[roberta_layer + ".self_attn_layer_norm.bias"]

        ### intermediate
        # intermediate: BertIntermediate = layer.intermediate
        intermediate = layer + ".intermediate"
        assert (
                model.roberta.encoder.layer[i].intermediate.dense.weight.shape == stat_dict[
            roberta_layer + ".fc1.weight"].shape
        )
        # TODO
        new_stat_dict[intermediate + ".dense.weight"] = stat_dict[roberta_layer + ".fc1.weight"]
        new_stat_dict[intermediate + ".dense.bias"] = stat_dict[roberta_layer + ".fc1.bias"]

        ### output
        # bert_output: BertOutput = layer.output
        bert_output = layer + ".output"
        assert (
                model.roberta.encoder.layer[i].output.dense.weight.shape == stat_dict[
            roberta_layer + ".fc2.weight"].shape
        )
        new_stat_dict[bert_output + ".dense.weight"] = stat_dict[roberta_layer + ".fc2.weight"]
        new_stat_dict[bert_output + ".dense.bias"] = stat_dict[roberta_layer + ".fc2.bias"]
        new_stat_dict[bert_output + ".LayerNorm.weight"] = stat_dict[roberta_layer + ".final_layer_norm.weight"]
        new_stat_dict[bert_output + ".LayerNorm.bias"] = stat_dict[roberta_layer + ".final_layer_norm.bias"]
        #### end of layer

    new_stat_dict["lm_head.dense.weight"] = stat_dict["decoder.lm_head.dense.weight"]
    new_stat_dict["lm_head.dense.bias"] = stat_dict["decoder.lm_head.dense.bias"]
    new_stat_dict["lm_head.layer_norm.weight"] = stat_dict["decoder.lm_head.layer_norm.weight"]
    new_stat_dict["lm_head.layer_norm.bias"] = stat_dict["decoder.lm_head.layer_norm.bias"]
    new_stat_dict["lm_head.decoder.weight"] = stat_dict["decoder.lm_head.weight"]
    new_stat_dict["lm_head.bias"] = stat_dict["decoder.lm_head.bias"]
    new_stat_dict["lm_head.decoder.bias"] = stat_dict["decoder.lm_head.bias"]

    new_stat_dict["roberta.pooler.dense.weight"] = model.roberta.pooler.dense.weight
    new_stat_dict["roberta.pooler.dense.bias"] = model.roberta.pooler.dense.bias

    return new_stat_dict


def update_hf_sd(old_sd, xlmr_path):
    x = torch.load(xlmr_path, map_location="cpu")
    m = old_sd
    d = OrderedDict()
    for k, v in m.items():
        if k == 'roberta.pooler.dense.weight':
            d[k] = x[k].half().clone()
        elif k not in ('proj_matrix_fast', 'lm_head.decoder.bias', 'roberta.pooler.dense.weight'):
            d[k] = v.data.half().clone()

    assert set(d.keys()) == set(x.keys())

    for k in d.keys():
        assert d[k].size() == x[k].size()

    for k in d.keys():
        if k != 'roberta.pooler.dense.weight':
            assert (d[k].float() - m[k].float()).abs().max().item() <= 1e-4

    return d


def convert_pt_to_hf(xlmr_path, inf, logger=None):
    if logger:
        logger.info("converting pt file at {} to hf file.".format(inf))
    sd = convert_cxlm_to_transformers(inf)
    return update_hf_sd(sd, xlmr_path)


if __name__ == "__main__":
    import os

    xlmr_path = "/home/v-zechi/data/unilm/zechi/exp/res/huggingface/hf-ckpt/xlmr-large/pytorch_model.bin"
    inf = "/home/v-zechi/data/unilm/zechi/exp/cxlm_exp/dump-ifx94-large/checkpoint_2_200000.pt"
    outf = "/home/v-zechi/data/unilm/zechi/usw/res/infoxlm-models/huggingface/infoxlm-large-without-meta/pytorch_model.bin"

    assert not os.path.exists(outf)
    sd = convert_cxlm_to_transformers(inf)
    sd2 = update_hf_sd(sd, xlmr_path)
    torch.save(sd2, outf)