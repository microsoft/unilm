import os
from typing import Optional
import logging

from fairseq.data import (
    IdDataset,
    NumSamplesDataset,
    NumelDataset,
    NestedDictionaryDataset,
    NumelDataset,
    RightPadDataset,
    RawLabelDataset,
)

from fairseq.tasks import register_task, FairseqDataclass, LegacyFairseqTask
from dataclasses import dataclass, field

from .data.tiktoken_tokenizer import TiktokenTokenizer
from .data.llama_tokenizer import LLaMATokenizer
from .data.utils import RawArrayDataset

from .harness_task import HarnessAnlir1, HarnessAnlir2, HarnessAnlir3, HarnessArc_challenge, HarnessArc_easy, HarnessBoolq, HarnessCopa, HarnessOpenbookqa, HarnessPiqa, HarnessRte, HarnessWic, HarnessWinogrande, HarnessHellaswag, HarnessRecord, HarnessTruthfullqaMC1, HarnessTruthfullqaMC2, HarnessSCIQ
from .harness_task import HarnessArc_challenge25s, HarnessHellaswag10s


logger = logging.getLogger(__name__)

task_map = {
    "harness_anli_r1": HarnessAnlir1,
    "harness_anli_r2": HarnessAnlir2,
    "harness_anli_r3": HarnessAnlir3,
    "harness_boolq": HarnessBoolq,
    "harness_copa": HarnessCopa,
    "harness_openbookqa": HarnessOpenbookqa,
    "harness_piqa": HarnessPiqa,
    "harness_rte": HarnessRte,
    "harness_wic": HarnessWic,
    "harness_winogrande": HarnessWinogrande,
    "harness_hellaswag": HarnessHellaswag,
    "harness_arc_challenge": HarnessArc_challenge,
    "harness_arc_easy": HarnessArc_easy,
    "harness_record": HarnessRecord,
    "harness_truthfullqa_mc1": HarnessTruthfullqaMC1,
    "harness_truthfullqa_mc2": HarnessTruthfullqaMC2,
    "harness_arc_challenge_25s": HarnessArc_challenge25s,
    "harness_hellaswag_10s": HarnessHellaswag10s,
    "harness_sciq": HarnessSCIQ,
}

from .mmlu_task import create_mmlu_tasks
mmlu_tasks = create_mmlu_tasks()
task_map.update(mmlu_tasks)

@dataclass
class HarnessEvalConfig(FairseqDataclass):
    data_dir: str = field(
        default="/mnt/msranlp/shaohanh/data/fs_eval/harness/",
        metadata={"help": "path to data directory"},
    )
    eval_data: str = field(default="", metadata={"help": "dataset name"})
    tokens_per_sample: int = field(
        default=2048,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    llama_model: Optional[str] = field(
        default=None,
        metadata={"help": "path to load tokenizer and config"},
    )
    tiktoken_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "tiktoken model to tokenize the data"
        },
    )
    tokenizer_pad_to_multiple: int = field(
        default=8,
        metadata={"help": "pad to multiple of this value"},
    )


@register_task('harness_eval', dataclass=HarnessEvalConfig)
class HarnessEval(LegacyFairseqTask):

    def __init__(self, cfg, tokenizer):
        super().__init__(cfg)
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.harness_task = task_map[self.cfg.eval_data](tokenizer=self.tokenizer, data_dir=cfg.data_dir, tokens_per_sample=cfg.tokens_per_sample)

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        if cfg.llama_model is not None:
            tokenizer = LLaMATokenizer(os.path.join(cfg.llama_model, "tokenizer.model"))
        elif cfg.tiktoken_model is not None:
            tokenizer = TiktokenTokenizer(cfg.tiktoken_model, cfg.tokenizer_pad_to_multiple)
        else:
            raise ValueError("No tokenizer model provided")

        return cls(cfg, tokenizer)

    def load_dataset(self, split, combine=False, **kwargs):
        src_tokens, gpt_loss_mask, label_length, labels = self.harness_task.get_data_for_evaluation()

        src_tokens = RawArrayDataset(src_tokens)
        gpt_loss_mask = RawArrayDataset(gpt_loss_mask, datatype="mask")
        label_length = RawLabelDataset(label_length)
        label_ids = RawLabelDataset(labels)
        '''
            Input format: src_tokens + option_tokens
        '''
        data_dict = {
            'id': IdDataset(),
            'net_input': {
                'src_tokens': RightPadDataset(
                    src_tokens,
                    pad_idx=self.tokenizer.pad_id,
                ),
                'gpt_loss_mask': RightPadDataset(
                    gpt_loss_mask,
                    pad_idx=False,
                ),
                'label_length': label_length,
                'src_lengths': NumelDataset(src_tokens, reduce=False),
            },
            'targets': label_ids,
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_tokens, reduce=True),
        }
        dataset = NestedDictionaryDataset(
            data_dict,
            sizes=[src_tokens.sizes],
        )

        print('| Loaded {} with {} samples'.format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    @property
    def target_dictionary(self):
        padding_idx = self.tokenizer.pad_id
        class Dict:
            def pad(self):
               return padding_idx
        dictionary = Dict()
        return dictionary

      