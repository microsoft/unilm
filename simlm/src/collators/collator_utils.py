import torch
import random
import warnings

from transformers import BertTokenizer, BertTokenizerFast, BatchEncoding
from typing import List, Union, Tuple, Any, Dict


def whole_word_mask(tokenizer: Union[BertTokenizer, BertTokenizerFast],
                    input_tokens: List[str],
                    mlm_prob: float,
                    max_predictions=512) -> List[int]:
    """
    Get 0/1 labels for masked tokens with whole word mask proxy
    """
    if not isinstance(tokenizer, (BertTokenizer, BertTokenizerFast)):
        warnings.warn(
            "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
            "Please refer to the documentation for more information."
        )

    cand_indexes = []
    for (i, token) in enumerate(input_tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue

        if len(cand_indexes) >= 1 and token.startswith("##"):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    random.shuffle(cand_indexes)
    num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * mlm_prob))))
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_lms.append(index)

    if len(covered_indexes) != len(masked_lms):
        raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
    mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
    return mask_labels


def torch_mask_tokens(tokenizer: Union[BertTokenizer, BertTokenizerFast],
                      inputs: torch.Tensor,
                      mask_labels: torch.Tensor,
                      all_use_mask_token: bool = False) -> Tuple[Any, Any]:
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
    'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
    """
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )
    labels = inputs.clone()
    masked_inputs = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

    probability_matrix = mask_labels.clone()

    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    masked_indices = probability_matrix.bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    if all_use_mask_token:
        masked_inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        return masked_inputs, labels

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    masked_inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    masked_inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return masked_inputs, labels


def merge_batch_dict(src_batch_dict: Union[Dict, BatchEncoding],
                     tgt_batch_dict: Union[Dict, BatchEncoding],
                     prefix: str = None):
    for key in src_batch_dict:
        tgt_batch_dict[(prefix or '') + key] = src_batch_dict[key].clone()
