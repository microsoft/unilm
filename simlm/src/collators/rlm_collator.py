import copy

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from transformers import BatchEncoding, BertTokenizerFast
from transformers.data.data_collator import _torch_collate_batch
from transformers.file_utils import PaddingStrategy

from config import Arguments
from .collator_utils import whole_word_mask, torch_mask_tokens, merge_batch_dict
from logger_config import logger


@dataclass
class DataCollatorForReplaceLM:
    tokenizer: BertTokenizerFast
    pad_to_multiple_of: Optional[int] = None
    args: Arguments = None

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(self, features: List[Dict]):
        return self.torch_call(features)

    def torch_call(self, examples: List[Dict[str, Any]]) -> BatchEncoding:
        if 'title' in examples[0]:
            text, text_pair = [ex['title'] for ex in examples], [ex['contents'] for ex in examples]
        else:
            text, text_pair = [ex['contents'] for ex in examples], None

        batch_dict = self.tokenizer(text,
                                    text_pair=text_pair,
                                    max_length=self.args.rlm_max_length,
                                    padding=PaddingStrategy.DO_NOT_PAD,
                                    truncation=True)

        encoder_mask_labels = []
        decoder_mask_labels = []
        extra_mlm_prob = self.args.rlm_decoder_mask_prob - self.args.rlm_encoder_mask_prob
        # mlm_prob + (1 - mlm_prob) x = decoder_prob
        # => x = (decoder_prob - mlm_prob) / (1 - mlm_prob)
        # since we mask twice independently, we need to adjust extra_mlm_prob accordingly
        extra_mlm_prob = extra_mlm_prob / (1 - self.args.rlm_encoder_mask_prob)

        for input_ids in batch_dict['input_ids']:
            ref_tokens = []
            for token_id in input_ids:
                token = self.tokenizer._convert_id_to_token(token_id)
                ref_tokens.append(token)
            encoder_mask_labels.append(whole_word_mask(self.tokenizer, ref_tokens,
                                                       mlm_prob=self.args.rlm_encoder_mask_prob))

            decoder_mask = encoder_mask_labels[-1][:]
            # overlapping mask
            if extra_mlm_prob > 1e-4:
                decoder_mask = [max(m1, m2) for m1, m2 in zip(decoder_mask,
                                whole_word_mask(self.tokenizer, ref_tokens, mlm_prob=extra_mlm_prob))]

            assert len(decoder_mask) == len(encoder_mask_labels[-1])
            decoder_mask_labels.append(decoder_mask)

        encoder_batch_mask = _torch_collate_batch(encoder_mask_labels, self.tokenizer,
                                                  pad_to_multiple_of=self.pad_to_multiple_of)

        encoder_batch_dict = self.tokenizer.pad(batch_dict,
                                                pad_to_multiple_of=self.pad_to_multiple_of,
                                                return_tensors="pt")
        encoder_inputs, encoder_labels = torch_mask_tokens(
            self.tokenizer, encoder_batch_dict['input_ids'], encoder_batch_mask,
            all_use_mask_token=self.args.all_use_mask_token)

        clean_input_ids = encoder_batch_dict['input_ids'].clone()
        encoder_batch_dict['input_ids'] = encoder_inputs
        encoder_batch_dict['labels'] = encoder_labels

        merged_batch_dict = BatchEncoding()
        merge_batch_dict(encoder_batch_dict, merged_batch_dict, prefix='enc_')

        decoder_batch_dict = copy.deepcopy(encoder_batch_dict)
        if extra_mlm_prob > 1e-4:
            decoder_batch_mask = _torch_collate_batch(decoder_mask_labels, self.tokenizer,
                                                      pad_to_multiple_of=self.pad_to_multiple_of)
            decoder_inputs, decoder_labels = torch_mask_tokens(
                self.tokenizer, clean_input_ids, decoder_batch_mask,
                all_use_mask_token=self.args.all_use_mask_token)

            decoder_batch_dict['input_ids'] = decoder_inputs
            decoder_batch_dict['labels'] = decoder_labels

        merge_batch_dict(decoder_batch_dict, merged_batch_dict, prefix='dec_')

        # simple integrity check
        # logger.info('encoder mask cnt: {}, decoder mask cnt: {}, non-equal input_ids cnt: {}'.format(
        #     (merged_batch_dict['enc_labels'] > 0).long().sum(),
        #     (merged_batch_dict['dec_labels'] > 0).long().sum(),
        #     (merged_batch_dict['dec_input_ids'] != merged_batch_dict['enc_input_ids']).long().sum()))

        labels = clean_input_ids.clone()
        for special_id in self.tokenizer.all_special_ids:
            labels[labels == special_id] = -100
        merged_batch_dict['labels'] = labels
        return merged_batch_dict
