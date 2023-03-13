# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
import numpy as np

import utils
from modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config


class TwoLayerMLP(nn.Module):
    def __init__(
            self, 
            in_features, 
            hidden_features, 
            out_features, 
            norm_layer, 
            norm_input=True, 
    ):
        super().__init__()
        self.norm1 = norm_layer(in_features) if norm_input else nn.Identity()
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.norm2 = norm_layer(hidden_features)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.norm1(x)
        x = self.dense1(x)
        x = self.norm2(x)
        x = self.act(x)
        return self.dense2(x)


class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x[:, 0, :]
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BEiT3ForVisualReasoning(BEiT3Wrapper):
    def __init__(
            self, 
            args, 
            num_classes, 
            norm_layer=nn.LayerNorm, 
            **kwargs
    ):
        super(BEiT3ForVisualReasoning, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.head = TwoLayerMLP(
            in_features=embed_dim * 4, 
            hidden_features=embed_dim * 2,
            out_features=num_classes, 
            norm_layer=norm_layer, 
        )
        init_scale = 0.001
        self.head.apply(self._init_weights)
        if isinstance(self.head.dense1, nn.Linear):
            self.head.dense1.weight.data.mul_(init_scale)
            self.head.dense1.bias.data.mul_(init_scale)

        if isinstance(self.head.dense2, nn.Linear):
            self.head.dense2.weight.data.mul_(init_scale)
            self.head.dense2.bias.data.mul_(init_scale)

    def forward(self, image_a, image_b, text_description, padding_mask, **kwargs):
        bsz, _ = text_description.size()
        
        vision_input = torch.cat((image_a, image_b), dim=0)
        language_input = torch.cat((text_description, text_description), dim=0)
        padding_mask = torch.cat((padding_mask, padding_mask), dim=0)

        outputs = self.beit3(
            textual_tokens=language_input, 
            visual_tokens=vision_input, 
            text_padding_position=padding_mask, 
        )
        x = outputs["encoder_out"]
        multiway_split_position = outputs["multiway_split_position"]

        vision_cls = x[:, 0, :]
        language_cls = x[:, multiway_split_position, :]
        cls_rep = torch.cat((vision_cls, language_cls), dim=-1)
        a, b = torch.split(cls_rep, split_size_or_sections=[bsz, bsz], dim=0)
        cls_rep = torch.cat((a, b), dim=-1)
        return self.head(cls_rep)
    

class BEiT3ForImageClassification(BEiT3Wrapper):
    def __init__(
            self, 
            args, 
            num_classes, 
            norm_layer=nn.LayerNorm, 
            **kwargs
    ):
        super(BEiT3ForImageClassification, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.fc_norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.fc_norm.apply(self._init_weights)
        self.head.apply(self._init_weights)
        init_scale = 0.001
        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def forward(self, image, **kwargs):
        x = self.beit3(textual_tokens=None, visual_tokens=image)["encoder_out"]
        t = x[:, 1:, :]
        cls_x = self.fc_norm(t.mean(1))
        return self.head(cls_x)


class BEiT3ForCaptioning(BEiT3Wrapper):
    def __init__(
            self, 
            args, 
            **kwargs
    ):
        super(BEiT3ForCaptioning, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.mlm_head = nn.Linear(embed_dim, args.vocab_size)
        self.mlm_head.apply(self._init_weights)

    def forward(self, image, text_ids, padding_mask, language_masked_pos, text_len=None, incremental_state=None, **kwargs):
        text_len = text_len if text_len is not None else text_ids.size(1)
        image_len = self.beit3.vision_embed.num_position_embeddings()
        max_len = text_len + image_len
        uni_mask = torch.zeros((max_len, max_len), dtype=torch.long, device=text_ids.device)
        i_start, i_end = 0, image_len
        t_start, t_end = image_len, max_len
        # triangle mask for caption to caption
        uni_mask[t_start:t_end, t_start:t_end] = torch.tril(torch.ones(text_len, text_len, dtype=torch.long, device=text_ids.device))
        # full attention for caption to image
        uni_mask[t_start:t_end, i_start:i_end] = 1
        # full attention for image to image
        uni_mask[i_start:i_end, i_start:i_end] = 1
        uni_mask = 1-uni_mask

        if incremental_state is not None:
            for idx in range(self.get_num_layers()):
                if idx not in incremental_state:
                    incremental_state[idx] = {}
        
        # for incremental decoding
        positions = None
        if image is None:
            uni_mask = uni_mask[-2:]
            padding_mask = None
            # start position (2 (fairseq starts at 2) + cur_position) is equal to text_len
            positions = torch.arange(text_len, text_ids.size(1) + text_len, device=text_ids.device).long().unsqueeze(0)

        outputs = self.beit3(
            textual_tokens=text_ids, 
            visual_tokens=image, 
            text_padding_position=padding_mask,
            attn_mask=uni_mask,
            incremental_state=incremental_state,
            positions=positions,
        )
        if image is not None:
            text_feats = outputs["encoder_out"][:, image_len:]
        else:
            text_feats = outputs["encoder_out"]

        if language_masked_pos is not None:
            text_feats = text_feats[language_masked_pos.bool()]

        return self.mlm_head(text_feats), incremental_state


class BEiT3ForVisualQuestionAnswering(BEiT3Wrapper):
    def __init__(
            self, 
            args, 
            num_classes, 
            norm_layer=nn.LayerNorm, 
            **kwargs
    ):
        super(BEiT3ForVisualQuestionAnswering, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.pooler = Pooler(
            input_features=embed_dim, 
            output_features=embed_dim, 
            norm_layer=norm_layer, 
        )
        self.pooler.apply(self._init_weights)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), 
            norm_layer(embed_dim * 2), 
            nn.GELU(), 
            nn.Linear(embed_dim * 2, num_classes), 
        )
        self.head.apply(self._init_weights)

    def forward(self, image, question, padding_mask, **kwargs):
        outputs = self.beit3(
            textual_tokens=question, 
            visual_tokens=image, 
            text_padding_position=padding_mask, 
        )
        x = outputs["encoder_out"]
        cls_rep = self.pooler(x)
        return self.head(cls_rep)


class BEiT3ForRetrieval(BEiT3Wrapper):
    def __init__(
            self, 
            args,
            **kwargs
    ):
        super(BEiT3ForRetrieval, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vision_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.language_head.apply(self._init_weights)
        self.vision_head.apply(self._init_weights)
        self.criterion = utils.ClipLoss(
            rank=utils.get_rank(), 
            world_size=utils.get_world_size(), 
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image=None, text_description=None, padding_mask=None, only_infer=False, **kwargs):
        if image is not None:
            outputs = self.beit3(
                textual_tokens=None, 
                visual_tokens=image, 
                text_padding_position=None, 
            )
            x = outputs["encoder_out"]
            vision_cls = self.vision_head(x[:, 0, :])
            vision_cls = F.normalize(vision_cls, dim=-1)
        else:
            vision_cls = None

        if text_description is not None:
            outputs = self.beit3(
                textual_tokens=text_description, 
                visual_tokens=None, 
                text_padding_position=padding_mask, 
            )
            x = outputs["encoder_out"]
            language_cls = self.language_head(x[:, 0, :])
            language_cls = F.normalize(language_cls, dim=-1)
        else:
            language_cls = None
        
        if only_infer:
            return vision_cls, language_cls
        else:
            loss, logits_per_image, logits_per_text = self.criterion(
                vision_cls, language_cls, self.logit_scale.exp())
            return loss, vision_cls, language_cls


@register_model
def beit3_base_patch16_224_imageclassification(pretrained=False, **kwargs):
    args = _get_base_config(**kwargs)
    args.normalize_output = False
    model = BEiT3ForImageClassification(args, num_classes=1000, **kwargs)
    return model


@register_model
def beit3_large_patch16_224_imageclassification(pretrained=False, **kwargs):
    args = _get_large_config(**kwargs)
    args.normalize_output = False
    model = BEiT3ForImageClassification(args, num_classes=1000, **kwargs)
    return model


@register_model
def beit3_base_patch16_224_nlvr2(pretrained=False, **kwargs):
    args = _get_base_config(**kwargs)
    model = BEiT3ForVisualReasoning(args, num_classes=2, **kwargs)
    return model


@register_model
def beit3_large_patch16_224_nlvr2(pretrained=False, **kwargs):
    args = _get_large_config(**kwargs)
    model = BEiT3ForVisualReasoning(args, num_classes=2, **kwargs)
    return model


@register_model
def beit3_base_patch16_384_vqav2(pretrained=False, **kwargs):
    args = _get_base_config(img_size=384, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129, **kwargs)
    return model


@register_model
def beit3_base_patch16_480_vqav2(pretrained=False, **kwargs):
    args = _get_base_config(img_size=480, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129, **kwargs)
    return model


@register_model
def beit3_large_patch16_384_vqav2(pretrained=False, **kwargs):
    args = _get_large_config(img_size=384, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129, **kwargs)
    return model


@register_model
def beit3_large_patch16_480_vqav2(pretrained=False, **kwargs):
    args = _get_large_config(img_size=480, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129, **kwargs)
    return model


@register_model
def beit3_large_patch16_768_vqav2(pretrained=False, **kwargs):
    args = _get_large_config(img_size=768, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129, **kwargs)
    return model


@register_model
def beit3_base_patch16_224_captioning(pretrained=False, **kwargs):
    args = _get_base_config(**kwargs)
    model = BEiT3ForCaptioning(args, **kwargs)
    return model


@register_model
def beit3_base_patch16_480_captioning(pretrained=False, **kwargs):
    args = _get_base_config(img_size=480, **kwargs)
    model = BEiT3ForCaptioning(args, **kwargs)
    return model


@register_model
def beit3_large_patch16_480_captioning(pretrained=False, **kwargs):
    args = _get_large_config(img_size=480, **kwargs)
    model = BEiT3ForCaptioning(args, **kwargs)
    return model


@register_model
def beit3_base_patch16_224_retrieval(pretrained=False, **kwargs):
    args = _get_base_config(**kwargs)
    model = BEiT3ForRetrieval(args, **kwargs)
    return model


@register_model
def beit3_base_patch16_384_retrieval(pretrained=False, **kwargs):
    args = _get_base_config(img_size=384, **kwargs)
    model = BEiT3ForRetrieval(args, **kwargs)
    return model


@register_model
def beit3_large_patch16_384_retrieval(pretrained=False, **kwargs):
    args = _get_large_config(img_size=384, **kwargs)
    model = BEiT3ForRetrieval(args, **kwargs)
    return model
