# ----------------------------------------------------------------------------
# SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data (https://arxiv.org/abs/2209.15329)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechLM
# Code based on fairseq: https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils, checkpoint_utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.transformer import Embedding
from fairseq.file_io import PathManager
from torch import Tensor
from fairseq.models.wav2vec.wav2vec2 import ConvFeatureExtractionModel
from fairseq.modules import GradMultiply, LayerNorm
from fairseq.tasks.hubert_pretraining import (
    HubertPretrainingConfig,
    HubertPretrainingTask,
)
from fairseq.models.hubert import HubertConfig
from fairseq.models.transformer import TransformerConfig
from speechlm.modules.w2v_encoder import TransformerEncoder
from speechlm.modules.transformer_encoder import TransformerEncoderBase

logger = logging.getLogger(__name__)

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])


@dataclass

class SpeechlmConfig(HubertConfig):
    use_rel_pos_enc: bool = field(
        default=False,
        metadata={"help": "whether to use relative positional encoding"},
    )
    scaling_for_att: float = field(
        default=1.0,
        metadata={"help": "scaling for attention weights to prevent overflow issue (for large model)"},
    )

    # unit encoder-decoder
    text_transformer: TransformerConfig = TransformerConfig()
    add_unit_encoder: bool = field(
        default=False,
        metadata={"help": "add unit encoder"},
    )
    add_decoder: bool = field(
        default=False,
        metadata={"help": "add decoder"},
    )
    add_text_ctc: bool = field(
        default=False,
        metadata={"help": "add_text_ctc head"},
    )
    text_ctc_conv_kernel: int = field(
        default=2,
        metadata={"help": "text_ctc_conv kernel size"},
    )
    mask_u2t: bool = field(
        default=True,
        metadata={"help": "mask the unit input in unit-to-text task"},
    )
    compute_mum: bool = field(
        default=False,
        metadata={"help": "compute MLM loss in unit-to-text task"},
    )

    # embedding mixing
    mix_with_unit: bool = field(
        default=True,
        metadata={"help": "mix with the unit embeddings"},
    )
    use_pred_unit: bool = field(
        default=False,
        metadata={"help": "use the embeddings of predicted units"},
    )
    l2_embedding: bool = field(
        default=False,
        metadata={"help": "compute l2 loss between unit embedding and unit hidden state"},
    )

    # Finetune related
    encoder_dict_size: int = field(
        default=-1,
        metadata={"help": "text encoder dictionary dimension"},
    )

    decoder_dict_size: int = field(
        default=-1,
        metadata={"help": "decoder dictionary dimension"},
    )


@register_model("speechlm", dataclass=SpeechlmConfig)
class SpeechlmModel(BaseFairseqModel):
    def __init__(
        self,
        cfg: SpeechlmConfig,
        task_cfg: HubertPretrainingConfig,
        dictionaries: List[Dictionary],
        unit_dictionary: Dictionary = None,
        text_tgt_dictionary: Dictionary = None,
    ) -> None:
        super().__init__()
        logger.info(f"SpeechlmModel Config: {cfg}")

        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )
        feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / task_cfg.sample_rate

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.final_dim = final_dim
        assert len(dictionaries) <= 2, f"Only support <=2 kinds of targets, get {len(dictionaries)} dictionaries"
        if len(dictionaries) == 1:
            dictionaries = [dictionaries[0], dictionaries[0]]
        
        self.final_proj_list = nn.ModuleList([
            nn.Linear(cfg.encoder_embed_dim, final_dim) for _ in dictionaries
        ])

        self.num_classes = [len(d) for d in dictionaries]
        self.label_embs_list = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(n, final_dim)) for n in self.num_classes
        ])
        for i in range(len(self.num_classes)):
            nn.init.uniform_(self.label_embs_list[i])

        ### build unit encoder:
        self.mask_u2t = cfg.mask_u2t
        self.compute_mum = cfg.compute_mum
        self.add_text_ctc = cfg.add_text_ctc
        self.text_ctc_conv_kernel = cfg.text_ctc_conv_kernel
        self.padding_idx = unit_dictionary.pad()
        self.unit_mask_idx = unit_dictionary.index("<mask>")

        self.add_unit_encoder = cfg.add_unit_encoder
        self.mix_with_unit = cfg.mix_with_unit
        self.use_pred_unit = cfg.use_pred_unit
        self.l2_embedding = cfg.l2_embedding
        if self.add_unit_encoder:
            assert len(unit_dictionary) == self.num_classes[0], f"unit_dictionary: {len(unit_dictionary)}, self.num_classes[0]: {self.num_classes[0]}"
            ### build unit pre-net, and shared with hubert label_embs if needed (default: False)
            self.unit_embed_tokens = self.build_embedding(
                    unit_dictionary,
                    cfg.text_transformer.encoder.embed_dim,
                )
            if self.final_dim == cfg.text_transformer.encoder.embed_dim:
                logger.info("Share label_embs[0] with unit_embed_tokens ...")
                nn.init.uniform_(self.unit_embed_tokens.weight)
                self.label_embs_list[0] = self.unit_embed_tokens.weight

            ### build unit encoder
            self.unit_encoder = TransformerEncoderBase(
                cfg.text_transformer, 
                unit_dictionary, 
                self.unit_embed_tokens,
                use_rel_pos_enc=cfg.use_rel_pos_enc,
                scaling_for_att=cfg.scaling_for_att,
            )
            
            ### build text ctc head
            if self.add_text_ctc:
                conv = nn.Conv1d(
                    cfg.text_transformer.encoder.embed_dim, cfg.text_transformer.encoder.embed_dim, 
                    self.text_ctc_conv_kernel,
                    stride=self.text_ctc_conv_kernel // 2,
                    bias=False,
                    padding=self.text_ctc_conv_kernel // 2,
                )
                nn.init.kaiming_normal_(conv.weight)
                self.unit_encoder_ctc_head = nn.Sequential(
                    Rotate3D(),
                    conv,
                    nn.Dropout(p=0.1),
                    nn.Sequential(
                        Rotate3D(),
                        Rotate3D(),
                        LayerNorm(cfg.text_transformer.encoder.embed_dim),
                    ),
                    nn.GELU(),
                    nn.Linear(cfg.text_transformer.encoder.embed_dim, len(text_tgt_dictionary)),
                )

        ### build unit2text decoder, not available for now
        self.add_decoder = cfg.add_decoder

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        return Embedding(num_embeddings, embed_dim, padding_idx)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: SpeechlmConfig, task: HubertPretrainingTask):
        """Build a new model instance."""
        unit_dictionary = getattr(task, "text_src_dictionary", None)
        text_tgt_dictionary = getattr(task, "text_dictionary", None)
        model = SpeechlmModel(cfg, task.cfg, task.dictionaries, unit_dictionary, text_tgt_dictionary)
        return model

    def apply_mask(self, x, padding_mask, target_list):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        return features

    def forward_targets(
        self,
        features: torch.Tensor,
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_inds += np.random.choice(int(self.feat2tar_ratio))
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, target_list

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def downsample_ctc_padding_mask(self, padding_mask):
        """
        padding_mask: (B, T)
        """
        stride = self.text_ctc_conv_kernel // 2
        return padding_mask[:, ::stride]
    
    def compute_pred(self, proj_x, label_embs):
        if self.target_glu:
            label_embs = self.target_glu(label_embs)
        x = F.normalize(proj_x.float(), dim=-1)                 # (S, D)
        label_embs = F.normalize(label_embs.float(), dim=-1)    # (C, D)
        logits = torch.matmul(x, label_embs.T).type_as(proj_x)  # (S, C)
        logits /= self.logit_temp
        return logits

    def compute_hubert_logits(self, x, target, proj, label_embs, padding_mask, mask_indices):
        if not self.skip_masked:
            masked_indices = torch.logical_and(~padding_mask, mask_indices)
            proj_x_m = proj(x[masked_indices])
            logit_m_list = [(self.compute_pred(proj_x_m, label_embs), target[masked_indices])]
        else:
            logit_m_list = [None]

        if not self.skip_nomask:
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
            proj_x_u = proj(x[nomask_indices])
            logit_u_list = [(self.compute_pred(proj_x_u, label_embs), target[nomask_indices])]
        else:
            logit_u_list = [None]

        return logit_m_list, logit_u_list

    def convert_embeddings(self,
        x,
        padding_mask,
        target=None,
        mask_indices=None,
        mix_with_unit=False,
        use_pred_unit=False,
        l2_embedding=False,
        remask=False
    ):
        """
        1. Mix with units if needed (default: True)
        2. Prepare for unit_encoder inputs
        Inputs:
            x, (B, T, D)
        Return:
            src_tokens, (B, T)
            soft_embeddings, (B, T, D)
            l2_loss, a loss
        """
        soft_embeddings = self.final_proj_list[0](x) if x.size(-1) == self.final_dim else x
        if padding_mask is None:
            padding_mask = soft_embeddings.new_zeros(soft_embeddings.size(0), soft_embeddings.size(1), dtype=torch.long)
        if use_pred_unit:
            src_tokens = self.compute_pred(self.final_proj_list[0](x), self.label_embs_list[0]).argmax(dim=-1)
            src_tokens[padding_mask] = self.padding_idx
        elif target is not None:
            src_tokens = target
        else:
            src_tokens = padding_mask.long()

        if l2_embedding | mix_with_unit:
            unit_embeddings = self.unit_embed_tokens(src_tokens)    # (B, T, D)
        
        l2_loss = 0
        if l2_embedding:
            if mask_indices is not None:
                l2_loss = (soft_embeddings - unit_embeddings)[mask_indices].float().pow(2).mean(dim=-1)
                scale = unit_embeddings[mask_indices].float().pow(2).sum(dim=-1)
            else:
                l2_loss = (soft_embeddings - unit_embeddings).float().pow(2).mean(dim=-1)
                scale = unit_embeddings.float().pow(2).sum(dim=-1)
            l2_loss = (l2_loss / scale).mean()

        if mix_with_unit:
            B, T, D = x.shape
            selected_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob / 2,
                self.mask_length // 2,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            selected_indices = torch.from_numpy(selected_indices).to(x.device)
            if mask_indices is not None:
                if remask:
                    remask_indices = torch.logical_and(selected_indices, mask_indices)
                    soft_embeddings[remask_indices] = self.mask_emb
                swap_indices = torch.logical_and(selected_indices, ~mask_indices)
            else:
                swap_indices = selected_indices
            soft_embeddings[swap_indices] = unit_embeddings[swap_indices]

        soft_embeddings = soft_embeddings * (1 - padding_mask.unsqueeze(-1).type_as(x))
        return src_tokens, soft_embeddings, l2_loss

    def forward(
        self,
        source: torch.Tensor = None,
        src_tokens: torch.Tensor = None,
        src_lengths: torch.Tensor = None,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        assert source is not None or src_tokens is not None
        if source is not None:
            return self.forward_speech(
                source=source,
                target_list=target_list,
                padding_mask=padding_mask,
                mask=mask,
                features_only=features_only,
                output_layer=output_layer,
            )
        else:
            return self.forward_text(
                src_tokens=src_tokens,
                src_lengths=src_lengths,
                mask=self.mask_u2t,
                output_layer=output_layer,
            )
    
    def forward_speech(
        self,
        source: torch.Tensor = None,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        features = self.forward_features(source)
        if target_list is not None:
            features, target_list = self.forward_targets(features, target_list)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask, target_list)
        else:
            x = features
            mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )

        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}
        
        logit_m_list, logit_u_list = self.compute_hubert_logits(
            x,
            target_list[0],
            self.final_proj_list[0],
            self.label_embs_list[0],
            padding_mask,
            mask_indices,
        )

        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }
        
        if self.add_unit_encoder:
            src_tokens, x_emb, l2_loss = self.convert_embeddings(
                x, 
                padding_mask, target_list[0],
                mask_indices=mask_indices,
                mix_with_unit=self.mix_with_unit,
                use_pred_unit=self.use_pred_unit,
                l2_embedding=self.l2_embedding,
            )
            encoder_out = self.unit_encoder(src_tokens, token_embeddings=x_emb)

            result['encoder_out'] = encoder_out['encoder_out']  # [(T, B, D)]
            result['encoder_padding_mask'] = encoder_out['encoder_padding_mask']    # [(B, T)]
            if self.l2_embedding:
                result['embedding_l2_loss'] = l2_loss

            code_logit_m_list, code_logit_u_list = self.compute_hubert_logits(
                encoder_out['encoder_out'][0].transpose(0, 1), 
                target_list[-1], 
                self.final_proj_list[-1], 
                self.label_embs_list[-1],
                padding_mask,
                mask_indices,
            )
            result['logit_m_list'] += code_logit_m_list
            result['logit_u_list'] += code_logit_u_list
        return result

    def forward_text(
        self,
        src_tokens: torch.Tensor = None,
        src_lengths: torch.Tensor = None,
        target_list: Optional[List[torch.Tensor]] = None,
        mask: bool = True,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        assert self.add_unit_encoder, f"Can not forward unit-text branch without unit_encoder!"

        padding_mask = src_tokens == self.padding_idx
        unit_embeddings = self.unit_embed_tokens(src_tokens)
        if mask:
            unit_embeddings, mask_indices = self.apply_mask(unit_embeddings, padding_mask, [src_tokens])
        else:
            ### If already applied mask on src_tokens, then the target_list should contains many padding_idx
            mask_indices = target_list[-1] != self.padding_idx
            unit_embeddings[mask_indices] = self.mask_emb
        
        encoder_out = self.unit_encoder(
            src_tokens,
            token_embeddings=unit_embeddings,
            return_all_hiddens=output_layer is not None,
        )

        result = {}
        result["encoder_out"] = encoder_out["encoder_out"]
        result["encoder_states"] = encoder_out["encoder_states"]
        result["padding_mask"] = padding_mask

        if self.compute_mum:
            code_logit_m_list, code_logit_u_list = self.compute_hubert_logits(
                encoder_out["encoder_out"].transpose(0, 1), 
                target_list[-1], 
                self.final_proj_list[-1], 
                self.label_embs_list[-1],
                padding_mask,
                mask_indices,
            )
            result["logit_m_list"] = code_logit_m_list
            result["logit_u_list"] = code_logit_u_list
        
        if self.add_text_ctc:
            result["encoder_out_ctc"] = [self.unit_encoder_ctc_head(x) for x in encoder_out['encoder_out']]
            result["encoder_padding_mask"] = [
                self.downsample_ctc_padding_mask(padding_mask) for padding_mask in encoder_out['encoder_padding_mask']
            ]
        return result

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features for only speech input"""
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        x = res["x"] # B x T x D
        padding_mask = res["padding_mask"]

        if self.add_unit_encoder:
            src_tokens, x, _ = self.convert_embeddings(
                x,
                padding_mask,
                mix_with_unit=False,
                use_pred_unit=False,
            )
            encoder_out = self.unit_encoder(
                src_tokens,
                token_embeddings=x,
                return_all_hiddens=output_layer is not None
            )
            res["x"] = encoder_out['encoder_out'][0].transpose(0, 1)  # (B, T, D)
        
        feature = res["features"] if ret_conv else res["x"]
        if output_layer is not None:
            feature = encoder_out['encoder_states']

        return feature, padding_mask

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x[0].float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        targets_list = [x[1].long() for x in logits_list if x is not None]
        return targets_list

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        if "embedding_l2_loss" in net_output:
            extra_losses.append(net_output["embedding_l2_loss"])
            names.append("embedding_l2_loss")

        return extra_losses, names

    def remove_pretraining_modules(self, step2=False):
        self.target_glu = None

    def load_checkpoint(self, checkpoint: str):
        if not PathManager.exists(checkpoint):
            raise IOError("Model file not found: {}".format(checkpoint))
        state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint)
        return state

class Rotate3D(nn.Module):
    """
    (T, B, D) --> (B, D, T) --> (D, T, B) --> (T, B, D)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(1, 2, 0)
