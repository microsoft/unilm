# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from speecht5.models.modules.speech_encoder_prenet import SpeechEncoderPrenet
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import GuidedAttentionLoss
from omegaconf import II
from typing import Any


@dataclass
class TexttoSpeechLossConfig(FairseqDataclass):
    use_masking: bool = field(
        default=True,
        metadata={"help": "Whether to use masking in calculation of loss"},
    )
    use_weighted_masking: bool = field(
        default=False,
        metadata={"help": "Whether to use weighted masking in calculation of loss"},
    )
    loss_type: str = field(
        default="L1",
        metadata={"help": "How to calc loss"},
    )
    bce_pos_weight: float = field(
        default=5.0,
        metadata={"help": "Positive sample weight in BCE calculation (only for use-masking=True)"},
    )
    bce_loss_lambda: float = field(
        default=1.0,
        metadata={"help": "Lambda in bce loss"},
    )
    use_guided_attn_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use guided attention loss"},
    )
    guided_attn_loss_sigma: float = field(
        default=0.4,
        metadata={"help": "Sigma in guided attention loss"},
    )
    guided_attn_loss_lambda: float = field(
        default=10.0,
        metadata={"help": "Lambda in guided attention loss"},
    )
    num_layers_applied_guided_attn: int = field(
        default=2,
        metadata={"help": "Number of layers to be applied guided attention loss, if set -1, all of the layers will be applied."},
    )
    num_heads_applied_guided_attn: int = field(
        default=2,
        metadata={"help": "Number of heads in each layer to be applied guided attention loss, if set -1, all of the heads will be applied."},
    )
    modules_applied_guided_attn: Any = field(
        default=("encoder-decoder",),
        metadata={"help": "Module name list to be applied guided attention loss"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


class TexttoSpeechLoss(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        use_masking=True,
        use_weighted_masking=False,
        loss_type="L1",
        bce_pos_weight=5.0,
        bce_loss_lambda=1.0,
        use_guided_attn_loss=False,
        guided_attn_loss_sigma=0.4,
        guided_attn_loss_lambda=1.0,
        num_layers_applied_guided_attn=2,
        num_heads_applied_guided_attn=2,
        modules_applied_guided_attn=["encoder-decoder"],
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking
        self.loss_type = loss_type
        self.bce_pos_weight = bce_pos_weight
        self.bce_loss_lambda = bce_loss_lambda
        self.use_guided_attn_loss = use_guided_attn_loss
        self.guided_attn_loss_sigma = guided_attn_loss_sigma
        self.guided_attn_loss_lambda = guided_attn_loss_lambda
        # define loss function
        self.criterion = Tacotron2Loss(
            use_masking=use_masking,
            use_weighted_masking=use_weighted_masking,
            bce_pos_weight=bce_pos_weight,
        )
        if self.use_guided_attn_loss:
            self.num_layers_applied_guided_attn = num_layers_applied_guided_attn
            self.num_heads_applied_guided_attn = num_heads_applied_guided_attn
            self.modules_applied_guided_attn = modules_applied_guided_attn
        if self.use_guided_attn_loss:
            self.attn_criterion = GuidedMultiHeadAttentionLoss(
                sigma=guided_attn_loss_sigma,
                alpha=guided_attn_loss_lambda,
            )

    def forward(self, model, sample):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, l1_loss, l2_loss, bce_loss, enc_dec_attn_loss = self.compute_loss(model, net_output, sample)
        # sample_size = (
        #     sample["target"].size(0) if self.sentence_avg else sample["nframes"]
        # )
        sample_size = 1
        logging_output = {
            "loss": loss.item(),
            "l1_loss": l1_loss.item(),
            "l2_loss": l2_loss.item(),
            "bce_loss": bce_loss.item(),
            "sample_size": 1,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
        }

        if enc_dec_attn_loss is not None:
            logging_output['enc_dec_attn_loss'] = enc_dec_attn_loss.item()

        if hasattr(model, 'text_encoder_prenet'):
            logging_output["encoder_alpha"] = model.text_encoder_prenet.encoder_prenet[-1].alpha.item()
            logging_output["decoder_alpha"] = model.speech_decoder_prenet.decoder_prenet[-1].alpha.item()
        elif hasattr(model, "speech_encoder_prenet"):
            logging_output["decoder_alpha"] = model.speech_decoder_prenet.decoder_prenet[-1].alpha.item()
        else:
            if 'task' not in sample:
                logging_output["encoder_alpha"] = model.encoder_prenet.encoder_prenet[-1].alpha.item()
            logging_output["decoder_alpha"] = model.decoder_prenet.decoder_prenet[-1].alpha.item()

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample):
        before_outs, after_outs, logits, attn = net_output
        labels = sample["labels"]
        ys = sample["dec_target"]
        olens = sample["dec_target_lengths"]
        ilens = sample["src_lengths"]

        # modifiy mod part of groundtruth
        if model.reduction_factor > 1:
            olens_in = olens.new([torch.div(olen, model.reduction_factor, rounding_mode='floor') for olen in olens])
            olens = olens.new([olen - olen % model.reduction_factor for olen in olens])
            max_olen = max(olens)
            ys = ys[:, :max_olen]
            labels = labels[:, :max_olen]
            labels = torch.scatter(labels, 1, (olens - 1).unsqueeze(1), 1.0) # make sure at least one frame has 1
            # labels[:, -1] = 1.0  
        else:
            olens_in = olens

        # caluculate loss values
        l1_loss, l2_loss, bce_loss = self.criterion(
            after_outs, before_outs, logits, ys, labels, olens
        )

        # l1_loss = l1_loss / ys.size(2)
        # l2_loss = l2_loss / ys.size(2)

        if self.loss_type == "L1":
            loss = l1_loss + self.bce_loss_lambda * bce_loss if self.bce_loss_lambda > 0.0 else l1_loss
        elif self.loss_type == "L2":
            loss = l2_loss + self.bce_loss_lambda * bce_loss if self.bce_loss_lambda > 0.0 else l2_loss
        elif self.loss_type == "L1+L2":
            loss = l1_loss + l2_loss + self.bce_loss_lambda * bce_loss if self.bce_loss_lambda > 0.0 else l1_loss + l2_loss
        else:
            raise ValueError("unknown --loss-type " + self.loss_type)

        # calculate guided attention loss
        enc_dec_attn_loss = None
        if self.use_guided_attn_loss:
            # calculate the input lengths of encoder, which is determined by encoder prenet
            if hasattr(model, 'encoder_reduction_factor') and model.encoder_reduction_factor > 1:
                ilens_in = ilens.new([ilen // model.encoder_reduction_factor for ilen in ilens])
            else:
                ilens_in = ilens
            # work for speech to speech model's input
            if "task_name" in sample and sample["task_name"] == "s2s":
                m = None
                if hasattr(model, 'encoder_prenet'):
                    m = model.encoder_prenet
                elif hasattr(model, 'speech_encoder_prenet'):
                    m = model.speech_encoder_prenet
                if m is not None and isinstance(m, SpeechEncoderPrenet):
                    ilens_in = m.get_src_lengths(ilens_in)
            # calculate for encoder-decoder
            if "encoder-decoder" in self.modules_applied_guided_attn:
                attn = [att_l[:, : self.num_heads_applied_guided_attn] for att_l in attn]
                att_ws = torch.cat(attn, dim=1)  # (B, H*L, T_out, T_in)
                enc_dec_attn_loss = self.attn_criterion(att_ws, ilens_in, olens_in)
                loss = loss + enc_dec_attn_loss

        return loss, l1_loss, l2_loss, bce_loss, enc_dec_attn_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        l1_loss_sum = sum(log.get("l1_loss", 0) for log in logging_outputs)
        l2_loss_sum = sum(log.get("l2_loss", 0) for log in logging_outputs)
        bce_loss_sum = sum(log.get("bce_loss", 0) for log in logging_outputs)
        sample_size = max(1, sum(log.get("sample_size", 0) for log in logging_outputs))
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, 1, round=5
        )
        encoder_alpha_sum = sum(log.get("encoder_alpha", 0) for log in logging_outputs)
        decoder_alpha_sum = sum(log.get("decoder_alpha", 0) for log in logging_outputs)
        ngpu = sum(log.get("ngpu", 0) for log in logging_outputs)

        metrics.log_scalar(
            "l1_loss", l1_loss_sum / sample_size, sample_size, 2, round=5
        )
        metrics.log_scalar(
            "l2_loss", l2_loss_sum / sample_size, sample_size, 2, round=5
        )
        metrics.log_scalar(
            "bce_loss", bce_loss_sum / sample_size, sample_size, 2, round=5
        )
        metrics.log_scalar(
            "encoder_alpha", encoder_alpha_sum / sample_size, sample_size, round=5
        )
        metrics.log_scalar(
            "decoder_alpha", decoder_alpha_sum / sample_size, sample_size, round=5
        )

        if "enc_dec_attn_loss" in logging_outputs[0]:
            enc_dec_attn_loss_sum = sum(log.get("enc_dec_attn_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "enc_dec_attn_loss", enc_dec_attn_loss_sum / sample_size, sample_size, round=8
            )


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

class Tacotron2Loss(torch.nn.Module):
    """Loss function module for Tacotron2."""

    def __init__(
        self, use_masking=True, use_weighted_masking=False, bce_pos_weight=20.0
    ):
        """Initialize Tactoron2 loss module.

        Args:
            use_masking (bool): Whether to apply masking
                for padded part in loss calculation.
            use_weighted_masking (bool):
                Whether to apply weighted masking in loss calculation.
            bce_pos_weight (float): Weight of positive sample of stop token.

        """
        super(Tacotron2Loss, self).__init__()
        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        # reduction = "none" if self.use_weighted_masking else "sum"
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(
            reduction=reduction, pos_weight=torch.tensor(bce_pos_weight)
        )

        # NOTE(kan-bayashi): register pre hook function for the compatibility
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def forward(self, after_outs, before_outs, logits, ys, labels, olens):
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            logits (Tensor): Batch of stop logits (B, Lmax).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            labels (LongTensor): Batch of the sequences of stop token labels (B, Lmax).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Mean square error loss value.
            Tensor: Binary cross entropy loss value.

        """
        # make mask and apply it
        if self.use_masking:
            masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            ys = ys.masked_select(masks)
            after_outs = after_outs.masked_select(masks)
            before_outs = before_outs.masked_select(masks)
            labels = labels.masked_select(masks[:, :, 0])
            logits = logits.masked_select(masks[:, :, 0])

        # calculate loss
        l1_loss = self.l1_criterion(after_outs, ys) + self.l1_criterion(before_outs, ys)
        mse_loss = self.mse_criterion(after_outs, ys) + self.mse_criterion(
            before_outs, ys
        )
        bce_loss = self.bce_criterion(logits, labels)

        # make weighted mask and apply it
        if self.use_weighted_masking:
            masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            weights = masks.float() / masks.sum(dim=1, keepdim=True).float()
            out_weights = weights.div(ys.size(0) * ys.size(2))
            logit_weights = weights.div(ys.size(0))

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(masks).sum()
            mse_loss = mse_loss.mul(out_weights).masked_select(masks).sum()
            bce_loss = (
                bce_loss.mul(logit_weights.squeeze(-1))
                .masked_select(masks.squeeze(-1))
                .sum()
            )

        return l1_loss, mse_loss, bce_loss

    def _load_state_dict_pre_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Apply pre hook fucntion before loading state dict.

        From v.0.6.1 `bce_criterion.pos_weight` param is registered as a parameter but
        old models do not include it and as a result, it causes missing key error when
        loading old model parameter. This function solve the issue by adding param in
        state dict before loading as a pre hook function
        of the `load_state_dict` method.

        """
        key = prefix + "bce_criterion.pos_weight"
        if key not in state_dict:
            state_dict[key] = self.bce_criterion.pos_weight

class GuidedMultiHeadAttentionLoss(GuidedAttentionLoss):
    """Guided attention loss function module for multi head attention.
    Args:
        sigma (float, optional): Standard deviation to control
        how close attention to a diagonal.
        alpha (float, optional): Scaling coefficient (lambda).
        reset_always (bool, optional): Whether to always reset masks.
    """

    def forward(self, att_ws, ilens, olens):
        """Calculate forward propagation.
        Args:
            att_ws (Tensor):
                Batch of multi head attention weights (B, H, T_max_out, T_max_in).
            ilens (LongTensor): Batch of input lenghts (B,).
            olens (LongTensor): Batch of output lenghts (B,).
        Returns:
            Tensor: Guided attention loss value.
        """
        if self.guided_attn_masks is None:
            self.guided_attn_masks = (
                self._make_guided_attention_masks(ilens, olens)
                .to(att_ws.device)
                .unsqueeze(1)
            )
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device).unsqueeze(1)
        losses = self.guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))
        if self.reset_always:
            self._reset_masks()

        return self.alpha * loss

    def _make_guided_attention_masks(self, ilens, olens):
        n_batches = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen), device=olens.device)
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(
                ilen, olen, self.sigma
            )
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma):
        grid_x, grid_y = torch.meshgrid(torch.arange(olen, device=olen.device), torch.arange(ilen, device=olen.device))
        grid_x, grid_y = grid_x.float(), grid_y.float()
        return 1.0 - torch.exp(
            -((grid_y / ilen - grid_x / olen) ** 2) / (2 * (sigma**2))
        )

    @staticmethod
    def _make_masks(ilens, olens):
        in_masks = make_non_pad_mask(ilens).to(ilens.device)  # (B, T_in)
        out_masks = make_non_pad_mask(olens).to(olens.device)  # (B, T_out)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)  # (B, T_out, T_in)
