import torch

from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import DataCollatorWithPadding, BatchEncoding


def _unpack_doc_values(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    doc_examples = []
    for f in features:
        keys = list(f.keys())
        lists_per_key = len(f[keys[0]])
        for idx in range(lists_per_key):
            doc_examples.append({k: f[k][idx] for k in keys})
    return doc_examples


@dataclass
class BiencoderCollator(DataCollatorWithPadding):

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        q_prefix, d_prefix = 'q_', 'd_'
        query_examples = [{k[len(q_prefix):]: v for k, v in f.items() if k.startswith(q_prefix)} for f in features]
        doc_examples = _unpack_doc_values(
            [{k[len(d_prefix):]: v for k, v in f.items() if k.startswith(d_prefix)} for f in features])
        assert len(doc_examples) % len(query_examples) == 0, \
            '{} doc and {} queries'.format(len(doc_examples), len(query_examples))

        # already truncated during tokenization
        q_collated = self.tokenizer.pad(
            query_examples,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors)
        d_collated = self.tokenizer.pad(
            doc_examples,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors)

        # merge into a single BatchEncoding by adding prefix
        for k in list(q_collated.keys()):
            q_collated[q_prefix + k] = q_collated[k]
            del q_collated[k]
        for k in d_collated:
            q_collated[d_prefix + k] = d_collated[k]

        merged_batch_dict = q_collated
        # dummy placeholder for field "labels", won't use it to compute loss
        labels = torch.zeros(len(query_examples), dtype=torch.long)
        merged_batch_dict['labels'] = labels

        if 'kd_labels' in features[0]:
            kd_labels = torch.stack([torch.tensor(f['kd_labels']) for f in features], dim=0).float()
            merged_batch_dict['kd_labels'] = kd_labels
        return merged_batch_dict
