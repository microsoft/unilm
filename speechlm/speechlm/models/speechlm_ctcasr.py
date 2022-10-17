# ----------------------------------------------------------------------------
# SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data (https://arxiv.org/abs/2209.15329)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechLM
# Code based on fairseq: https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

from dataclasses import dataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.tasks import FairseqTask

from fairseq.models.hubert import HubertAsrConfig, HubertCtc, HubertEncoder

@dataclass
class SpeechLMCtcConfig(HubertAsrConfig):
    pass


@register_model("speechlm_ctc", dataclass=SpeechLMCtcConfig)
class SpeechLMCtc(HubertCtc):
    def __init__(self, cfg: SpeechLMCtcConfig, w2v_encoder: BaseFairseqModel):
        super().__init__(cfg, w2v_encoder)

    @classmethod
    def build_model(cls, cfg: SpeechLMCtcConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = SpeechLMEncoder(cfg, task)
        return cls(cfg, w2v_encoder)


class SpeechLMEncoder(HubertEncoder):
    def __init__(self, cfg: HubertAsrConfig, task):
        super().__init__(cfg, task)
        
        if (task.target_dictionary is not None) and (
            hasattr(self.w2v_model, "unit_encoder_ctc_head")
        ):
            self.proj = self.w2v_model.unit_encoder_ctc_head
            self.conv_ctc_proj = True
        else:
            self.conv_ctc_proj = False

    def forward(self, source, padding_mask, tbc=True, **kwargs):
        results = super().forward(
            source,
            padding_mask,
            tbc,
            **kwargs,
        )
        if self.conv_ctc_proj:
            results["padding_mask"] = self.w2v_model.downsample_ctc_padding_mask(results["padding_mask"])
        return results
