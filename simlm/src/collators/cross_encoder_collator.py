import torch

from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import BatchEncoding, DataCollatorWithPadding


@dataclass
class CrossEncoderCollator(DataCollatorWithPadding):

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        unpack_features = []
        for ex in features:
            keys = list(ex.keys())
            # assert all(len(ex[k]) == 8 for k in keys)
            for idx in range(len(ex[keys[0]])):
                unpack_features.append({k: ex[k][idx] for k in keys})

        collated_batch_dict = self.tokenizer.pad(
            unpack_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors)

        collated_batch_dict['labels'] = torch.zeros(len(features), dtype=torch.long)
        return collated_batch_dict
