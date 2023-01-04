import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from torch import Tensor
from transformers import (
    AutoModel,
    PreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput

from config import Arguments
from logger_config import logger
from utils import dist_gather_tensor, select_grouped_indices, full_contrastive_scores_and_labels


@dataclass
class BiencoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    labels: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BiencoderModel(nn.Module):
    def __init__(self, args: Arguments,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel):
        super().__init__()
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.args = args
        self.pooler = nn.Linear(self.lm_q.config.hidden_size, args.out_dimension) if args.add_pooler else nn.Identity()

        from trainers import BiencoderTrainer
        self.trainer: Optional[BiencoderTrainer] = None

    def forward(self, query: Dict[str, Tensor] = None,
                passage: Dict[str, Tensor] = None):
        assert self.args.process_index >= 0

        scores, labels, q_reps, p_reps, all_scores, all_labels = self._compute_scores(query, passage)

        start = self.args.process_index * q_reps.shape[0]
        group_indices = select_grouped_indices(scores=scores,
                                               group_size=self.args.train_n_passages,
                                               start=start * self.args.train_n_passages)

        if not self.args.do_kd_biencoder:
            # training biencoder from scratch
            if self.args.use_scaled_loss:
                loss = self.cross_entropy(all_scores, all_labels)
                loss *= self.args.world_size if self.args.loss_scale <= 0 else self.args.loss_scale
            else:
                loss = self.cross_entropy(scores, labels)
        else:
            # training biencoder with kd
            # batch_size x train_n_passage
            group_scores = torch.gather(input=scores, dim=1, index=group_indices)
            assert group_scores.shape[1] == self.args.train_n_passages
            group_log_scores = torch.log_softmax(group_scores, dim=-1)
            kd_log_target = torch.log_softmax(query['kd_labels'], dim=-1)

            kd_loss = self.kl_loss_fn(input=group_log_scores, target=kd_log_target)

            # (optionally) mask out hard negatives
            if self.training and self.args.kd_mask_hn:
                scores = torch.scatter(input=scores, dim=1, index=group_indices[:, 1:], value=float('-inf'))
            if self.args.use_scaled_loss:
                ce_loss = self.cross_entropy(all_scores, all_labels)
                ce_loss *= self.args.world_size if self.args.loss_scale <= 0 else self.args.loss_scale
            else:
                ce_loss = self.cross_entropy(scores, labels)

            loss = self.args.kd_cont_loss_weight * ce_loss + kd_loss

        total_n_psg = self.args.world_size * q_reps.shape[0] * self.args.train_n_passages

        return BiencoderOutput(loss=loss, q_reps=q_reps, p_reps=p_reps,
                               labels=labels.contiguous(),
                               scores=scores[:, :total_n_psg].contiguous())

    def _compute_scores(self, query: Dict[str, Tensor] = None,
                        passage: Dict[str, Tensor] = None) -> Tuple:
        q_reps = self._encode(self.lm_q, query)
        p_reps = self._encode(self.lm_p, passage)

        all_q_reps = dist_gather_tensor(q_reps)
        all_p_reps = dist_gather_tensor(p_reps)
        assert all_p_reps.shape[0] == self.args.world_size * q_reps.shape[0] * self.args.train_n_passages

        all_scores, all_labels = full_contrastive_scores_and_labels(
            query=all_q_reps, key=all_p_reps,
            use_all_pairs=self.args.full_contrastive_loss)

        if self.args.l2_normalize:
            if self.args.t_warmup:
                scale = 1 / self.args.t * min(1.0, self.trainer.state.global_step / self.args.warmup_steps)
                scale = max(1.0, scale)
            else:
                scale = 1 / self.args.t
            all_scores = all_scores * scale

        start = self.args.process_index * q_reps.shape[0]
        local_query_indices = torch.arange(start, start + q_reps.shape[0], dtype=torch.long).to(q_reps.device)
        # batch_size x (world_size x batch_size x train_n_passage)
        scores = all_scores.index_select(dim=0, index=local_query_indices)
        labels = all_labels.index_select(dim=0, index=local_query_indices)

        return scores, labels, q_reps, p_reps, all_scores, all_labels

    def _encode(self, encoder: PreTrainedModel, input_dict: dict) -> Optional[torch.Tensor]:
        if not input_dict:
            return None
        outputs = encoder(**{k: v for k, v in input_dict.items() if k not in ['kd_labels']}, return_dict=True)
        hidden_state = outputs.last_hidden_state
        embeds = hidden_state[:, 0]
        embeds = self.pooler(embeds)
        if self.args.l2_normalize:
            embeds = F.normalize(embeds, dim=-1)
        return embeds.contiguous()

    @classmethod
    def build(cls, args: Arguments, **hf_kwargs):
        # load local
        if os.path.isdir(args.model_name_or_path):
            if not args.share_encoder:
                _qry_model_path = os.path.join(args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(args.model_name_or_path, 'passage_model')
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = args.model_name_or_path
                    _psg_model_path = args.model_name_or_path
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = AutoModel.from_pretrained(_qry_model_path, **hf_kwargs)
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = AutoModel.from_pretrained(_psg_model_path, **hf_kwargs)
            else:
                logger.info(f'loading shared model weight from {args.model_name_or_path}')
                lm_q = AutoModel.from_pretrained(args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = AutoModel.from_pretrained(args.model_name_or_path, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if not args.share_encoder else lm_q

        model = cls(args=args, lm_q=lm_q, lm_p=lm_p)
        return model

    def save(self, output_dir: str):
        if not self.args.share_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'passage_model'), exist_ok=True)
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'))
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'))
        else:
            self.lm_q.save_pretrained(output_dir)
        if self.args.add_pooler:
            torch.save(self.pooler.state_dict(), os.path.join(output_dir, 'pooler.pt'))


class BiencoderModelForInference(BiencoderModel):
    def __init__(self, args: Arguments,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel):
        nn.Module.__init__(self)
        self.args = args
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = nn.Linear(self.lm_q.config.hidden_size, args.out_dimension) if args.add_pooler else nn.Identity()

    @torch.no_grad()
    def forward(self, query: Dict[str, Tensor] = None,
                passage: Dict[str, Tensor] = None):
        q_reps = self._encode(self.lm_q, query)
        p_reps = self._encode(self.lm_p, passage)
        return BiencoderOutput(q_reps=q_reps, p_reps=p_reps)

    @classmethod
    def build(cls, args: Arguments, **hf_kwargs):
        model_name_or_path = args.model_name_or_path

        # load local
        if os.path.isdir(model_name_or_path):
            _qry_model_path = os.path.join(model_name_or_path, 'query_model')
            _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
            if os.path.exists(_qry_model_path):
                logger.info(f'found separate weight for query/passage encoders')
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = AutoModel.from_pretrained(_qry_model_path, **hf_kwargs)
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = AutoModel.from_pretrained(_psg_model_path, **hf_kwargs)
            else:
                logger.info(f'try loading tied weight')
                logger.info(f'loading model weight from {model_name_or_path}')
                lm_q = AutoModel.from_pretrained(model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        else:
            logger.info(f'try loading tied weight {model_name_or_path}')
            lm_q = AutoModel.from_pretrained(model_name_or_path, **hf_kwargs)
            lm_p = lm_q

        model = cls(args=args, lm_q=lm_q, lm_p=lm_p)

        pooler_path = os.path.join(args.model_name_or_path, 'pooler.pt')
        if os.path.exists(pooler_path):
            logger.info('loading pooler weights from local files')
            state_dict = torch.load(pooler_path, map_location="cpu")
            model.pooler.load_state_dict(state_dict)
        else:
            assert not args.add_pooler
            logger.info('No pooler will be loaded')
        return model
