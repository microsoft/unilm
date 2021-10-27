import torch
import logging

from transformers.modeling_utils import cached_path, WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME

logger = logging.getLogger(__name__)


def get_checkpoint_from_transformer_cache(
        archive_file, pretrained_model_name_or_path, pretrained_model_archive_map,
        cache_dir, force_download, proxies, resume_download,
):
    try:
        resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir, force_download=force_download,
                                            proxies=proxies, resume_download=resume_download)
    except EnvironmentError:
        if pretrained_model_name_or_path in pretrained_model_archive_map:
            msg = "Couldn't reach server at '{}' to download pretrained weights.".format(
                archive_file)
        else:
            msg = "Model name '{}' was not found in model name list ({}). " \
                  "We assumed '{}' was a path or url to model weight files named one of {} but " \
                  "couldn't find any such file at this path or url.".format(
                pretrained_model_name_or_path,
                ', '.join(pretrained_model_archive_map.keys()),
                archive_file,
                [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME])
        raise EnvironmentError(msg)

    if resolved_archive_file == archive_file:
        logger.info("loading weights file {}".format(archive_file))
    else:
        logger.info("loading weights file {} from cache at {}".format(
            archive_file, resolved_archive_file))

    return torch.load(resolved_archive_file, map_location='cpu')


def hf_roberta_to_hf_bert(state_dict):
    logger.info(" * Convert Huggingface RoBERTa format to Huggingface BERT format * ")

    new_state_dict = {}

    for key in state_dict:
        value = state_dict[key]
        if key == 'roberta.embeddings.position_embeddings.weight':
            value = value[2:]
        if key == 'roberta.embeddings.token_type_embeddings.weight':
            continue
        if key.startswith('roberta'):
            key = 'bert.' + key[8:]
        elif key.startswith('lm_head'):
            if 'layer_norm' in key or 'dense' in key:
                key = 'cls.predictions.transform.' + key[8:]
            else:
                key = 'cls.predictions.' + key[8:]
            key = key.replace('layer_norm', 'LayerNorm')

        new_state_dict[key] = value

    return new_state_dict


def hf_electra_to_hf_bert(state_dict):
    logger.info(" * Convert Huggingface ELECTRA format to Huggingface BERT format * ")

    new_state_dict = {}

    for key in state_dict:
        value = state_dict[key]
        if key.startswith('electra'):
            key = 'bert.' + key[8:]
        new_state_dict[key] = value

    return new_state_dict


def hf_bert_to_hf_bert(state_dict):
    # keep no change
    return state_dict


def unilm_to_hf_bert(state_dict):
    logger.info(" * Convert Fast QKV format to Huggingface BERT format * ")

    new_state_dict = {}

    for key in state_dict:
        value = state_dict[key]
        if key.endswith("attention.self.q_bias"):
            new_state_dict[key.replace("attention.self.q_bias", "attention.self.query.bias")] = value.view(-1)
        elif key.endswith("attention.self.v_bias"):
            new_state_dict[key.replace("attention.self.v_bias", "attention.self.value.bias")] = value.view(-1)
            new_state_dict[key.replace("attention.self.v_bias", "attention.self.key.bias")] = torch.zeros_like(value.view(-1))
        elif key.endswith("attention.self.qkv_linear.weight"):
            l, _ = value.size()
            assert l % 3 == 0
            l = l // 3
            q, k, v = torch.split(value, split_size_or_sections=(l, l, l), dim=0)
            new_state_dict[key.replace("attention.self.qkv_linear.weight", "attention.self.query.weight")] = q
            new_state_dict[key.replace("attention.self.qkv_linear.weight", "attention.self.key.weight")] = k
            new_state_dict[key.replace("attention.self.qkv_linear.weight", "attention.self.value.weight")] = v
        elif key == "bert.encoder.rel_pos_bias.weight":
            new_state_dict["bert.rel_pos_bias.weight"] = value
        else:
            new_state_dict[key] = value

    del state_dict

    return new_state_dict


state_dict_convert = {
    'bert': hf_bert_to_hf_bert,
    'unilm': unilm_to_hf_bert, 
    'minilm': hf_bert_to_hf_bert, 
    'roberta': hf_roberta_to_hf_bert,
    'xlm-roberta': hf_roberta_to_hf_bert,
    'electra': hf_electra_to_hf_bert,
}
