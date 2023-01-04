import torch
import torch.nn as nn

from typing import Optional, Dict
from transformers import (
    PreTrainedModel,
    AutoModelForSequenceClassification
)
from transformers.modeling_outputs import SequenceClassifierOutput

from config import Arguments


class Reranker(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, args: Arguments):
        super().__init__()
        self.hf_model = hf_model
        self.args = args

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, batch: Dict[str, torch.Tensor]) -> SequenceClassifierOutput:
        input_batch_dict = {k: v for k, v in batch.items() if k != 'labels'}

        if self.args.rerank_forward_factor > 1:
            assert torch.sum(batch['labels']).long().item() == 0
            assert all(len(v.shape) == 2 for v in input_batch_dict.values())

            is_train = self.hf_model.training
            self.hf_model.eval()

            with torch.no_grad():
                outputs: SequenceClassifierOutput = self.hf_model(**input_batch_dict, return_dict=True)
                outputs.logits = outputs.logits.view(-1, self.args.train_n_passages)
                # make sure the target passage is not masked out
                outputs.logits[:, 0].fill_(float('inf'))

                k = self.args.train_n_passages // self.args.rerank_forward_factor
                _, topk_indices = torch.topk(outputs.logits, k=k, dim=-1, largest=True)
                topk_indices += self.args.train_n_passages * torch.arange(0, topk_indices.shape[0],
                                                                          dtype=torch.long,
                                                                          device=topk_indices.device).unsqueeze(-1)
                topk_indices = topk_indices.view(-1)

                input_batch_dict = {k: v.index_select(dim=0, index=topk_indices) for k, v in input_batch_dict.items()}

            self.hf_model.train(is_train)

        n_psg_per_query = self.args.train_n_passages // self.args.rerank_forward_factor

        if self.args.rerank_use_rdrop and self.training:
            input_batch_dict = {k: torch.cat([v, v], dim=0) for k, v in input_batch_dict.items()}

        outputs: SequenceClassifierOutput = self.hf_model(**input_batch_dict, return_dict=True)

        if self.args.rerank_use_rdrop and self.training:
            logits = outputs.logits.view(2, -1, n_psg_per_query)
            outputs.logits = logits[0, :, :].contiguous()
            log_prob = torch.log_softmax(logits, dim=2)
            log_prob1, log_prob2 = log_prob[0, :, :], log_prob[1, :, :]
            rdrop_loss = 0.5 * (self.kl_loss_fn(log_prob1, log_prob2) + self.kl_loss_fn(log_prob2, log_prob1))
            ce_loss = 0.5 * (self.cross_entropy(log_prob1, batch['labels'])
                             + self.cross_entropy(log_prob2, batch['labels']))

            outputs.loss = rdrop_loss + ce_loss
        else:
            outputs.logits = outputs.logits.view(-1, n_psg_per_query)
            loss = self.cross_entropy(outputs.logits, batch['labels'])
            outputs.loss = loss

        return outputs

    @classmethod
    def from_pretrained(cls, all_args: Arguments, *args, **kwargs):
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        return cls(hf_model, all_args)

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)


class RerankerForInference(nn.Module):
    def __init__(self, hf_model: Optional[PreTrainedModel] = None):
        super().__init__()
        self.hf_model = hf_model
        self.hf_model.eval()

    @torch.no_grad()
    def forward(self, batch) -> SequenceClassifierOutput:
        return self.hf_model(**batch)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        hf_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)
        return cls(hf_model)
