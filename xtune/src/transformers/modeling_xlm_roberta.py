# coding=utf-8
# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch XLM-RoBERTa model. """

import logging

import torch.nn as nn
from .configuration_xlm_roberta import XLMRobertaConfig
from .file_utils import add_start_docstrings
from .modeling_roberta import (
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForSequenceClassification,
    RobertaForMultiTaskSequenceClassification,
    RobertaForTokenClassification,
    RobertaForQuestionAnswering,
    RobertaModel,
)

logger = logging.getLogger(__name__)

XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "xlm-roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-base-pytorch_model.bin",
    "xlm-roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll02-dutch": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-dutch-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll02-spanish": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-spanish-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll03-english": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-english-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll03-german": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-german-pytorch_model.bin",
}

XLM_ROBERTA_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.XLMRobertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""


@add_start_docstrings(
    "The bare XLM-RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaModel(RobertaModel):
    """
    This class overrides :class:`~transformers.RobertaModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


@add_start_docstrings(
    """XLM-RoBERTa Model with a `language modeling` head on top. """, XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForMaskedLM(RobertaForMaskedLM):
    """
    This class overrides :class:`~transformers.RobertaForMaskedLM`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


@add_start_docstrings(
    """XLM-RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer
    on top of the pooled output) e.g. for GLUE tasks. """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForMultiTaskSequenceClassification(RobertaForMultiTaskSequenceClassification):
    """
    This class overrides :class:`~transformers.RobertaForSequenceClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


@add_start_docstrings(
    """XLM-RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer
    on top of the pooled output) e.g. for GLUE tasks. """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForSequenceClassification(RobertaForSequenceClassification):
    """
    This class overrides :class:`~transformers.RobertaForSequenceClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


@add_start_docstrings(
    """XLM-RoBERTa Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForMultipleChoice(RobertaForMultipleChoice):
    """
    This class overrides :class:`~transformers.RobertaForMultipleChoice`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


@add_start_docstrings(
    """XLM-RoBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForQuestionAnswering(RobertaForQuestionAnswering):
    """
    This class overrides :class:`~transformers.RobertaForQuestionAnswering`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


@add_start_docstrings(
    """XLM-RoBERTa Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForTokenClassification(RobertaForTokenClassification):
    """
    This class overrides :class:`~transformers.RobertaForTokenClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


from .modeling_bert import BertPreTrainedModel
from .modeling_roberta import RobertaClassificationHead, ROBERTA_INPUTS_DOCSTRING
from .file_utils import add_start_docstrings_to_callable
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


@add_start_docstrings(
    """XLM-RoBERTa Model transformer for cross-lingual retrieval""",
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForRetrieval(BertPreTrainedModel):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)

    @add_start_docstrings_to_callable(XLM_ROBERTA_START_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None
    ):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        outputs = (outputs[0], None, outputs[2])
        return outputs  # (loss), (hidden_states), (attentions)


def KL(input, target):
    input = input.float()
    target = target.float()
    loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target, dim=-1, dtype=torch.float32))
    return loss


@add_start_docstrings(
    """XLM-RoBERTa Model for cross-lingual classification (stabletune). """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForSequenceClassificationStable(BertPreTrainedModel):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, noised_data_generator=None):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

        self.noise_sampler = None
        self.enable_r1_loss = False
        self.original_loss = True
        self.noised_loss = False
        self.use_hard_labels = False
        if noised_data_generator is not None:
            self.enable_r1_loss = noised_data_generator.enable_r1_loss
            self.r1_lambda = noised_data_generator.r1_lambda
            self.r2_lambda = noised_data_generator.r2_lambda
            self.original_loss = noised_data_generator.original_loss
            self.noised_loss = noised_data_generator.noised_loss
            self.use_hard_labels = noised_data_generator.use_hard_labels
            self.augment_method = None
            self.enable_random_noise = noised_data_generator.enable_random_noise

            if noised_data_generator.enable_random_noise or self.augment_method == "gn":
                self.noise_detach_embeds = noised_data_generator.noise_detach_embeds
                if noised_data_generator.noise_type in {"normal"}:
                    self.noise_sampler = torch.distributions.normal.Normal(
                        loc=0.0, scale=noised_data_generator.noise_eps
                    )
                elif noised_data_generator.noise_type == "uniform":
                    self.noise_sampler = torch.distributions.uniform.Uniform(
                        low=-noised_data_generator.noise_eps, high=noised_data_generator.noise_eps
                    )
                else:
                    raise Exception(f"unrecognized noise type {noised_data_generator.noise_type}")

    @add_start_docstrings_to_callable(XLM_ROBERTA_START_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            noised_input_ids=None,
            noised_attention_mask=None,
            noised_token_type_ids=None,
            return_sequence_output=False,
            is_augmented=None,
            r1_mask=None,
            first_stage_model_logits=None,
    ):
        word_embeds = self.roberta.embeddings.word_embeddings(input_ids)

        if is_augmented is not None:
            # ground truth data indices
            # optional when data augmentation is considered noisy
            # if self.augment_method != "mt" and first_stage_model_logits is not None:
            #     gt_indices = (~is_augmented.bool()).view(-1).nonzero(as_tuple=False).view(-1).tolist()
            # else:
            #     gt_indices = list(range(0, input_ids.size(0)))

            gt_indices = list(range(0, input_ids.size(0)))
            augmented_indices = is_augmented.view(-1).nonzero(as_tuple=False).view(-1).tolist()
        else:
            gt_indices = list(range(0, input_ids.size(0)))
            augmented_indices = None

        if is_augmented is not None and self.augment_method == "gn":
            noise = self.noise_sampler.sample(sample_shape=word_embeds.shape).to(word_embeds)
            if self.noise_detach_embeds:
                noised_word_embeds = word_embeds.detach().clone() + noise
            else:
                noised_word_embeds = word_embeds + noise
            if len(augmented_indices) > 0:
                word_embeds[augmented_indices] = noised_word_embeds[augmented_indices]

        outputs = self.roberta(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=word_embeds,
        )

        sequence_output = outputs[0]
        if return_sequence_output:
            return sequence_output

        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if len(gt_indices) > 0:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1)[gt_indices], labels.view(-1)[gt_indices])
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels)[gt_indices], labels.view(-1)[gt_indices])
            else:
                loss_fct = CrossEntropyLoss()
                loss = logits.data.new([0.0])

            outputs = (loss,) + outputs

            if self.training:
                if first_stage_model_logits is not None and is_augmented is not None and len(augmented_indices) > 0:
                    # optional to use hard labels and augmented indices when data augmentation is considered noisy
                    if self.use_hard_labels:
                        hard_labels = first_stage_model_logits.view(-1, self.num_labels).max(dim=-1)[1]
                        r2_loss = loss_fct(logits.view(-1, self.num_labels)[augmented_indices],
                                           hard_labels.view(-1)[augmented_indices])
                    else:
                        r2_loss = KL(logits.view(-1, self.num_labels)[augmented_indices],
                                     first_stage_model_logits.view(-1, self.num_labels).detach()[augmented_indices])
                    r2_loss = r2_loss * self.r2_lambda
                else:
                    r2_loss = loss.data.new([0.0])

                if self.enable_r1_loss or self.noised_loss:
                    if noised_input_ids is not None:
                        noised_word_embeds = self.roberta.embeddings.word_embeddings(noised_input_ids)
                        assert noised_attention_mask is not None
                    elif self.noise_sampler is not None:
                        noised_word_embeds = self.roberta.embeddings.word_embeddings(input_ids)
                        noised_attention_mask = attention_mask
                        noised_token_type_ids = token_type_ids
                    else:
                        assert False

                    if self.enable_random_noise:
                        noise = self.noise_sampler.sample(sample_shape=noised_word_embeds.shape).to(noised_word_embeds)
                        if self.noise_detach_embeds:
                            noised_word_embeds = noised_word_embeds.detach().clone() + noise
                        else:
                            noised_word_embeds = noised_word_embeds + noise

                    noised_outputs = self.roberta(
                        input_ids=None,
                        attention_mask=noised_attention_mask,
                        token_type_ids=noised_token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=noised_word_embeds,
                    )
                    noised_sequence_output = noised_outputs[0]
                    noised_logits = self.classifier(noised_sequence_output)

                if self.original_loss:
                    original_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                else:
                    original_loss = loss.data.new([0.0])

                if self.noised_loss:
                    noised_loss = loss_fct(noised_logits.view(-1, self.num_labels), labels.view(-1))
                else:
                    noised_loss = loss.data.new([0.0])

                if self.enable_r1_loss and r1_mask.sum() > 0:
                    logits = logits.masked_select(r1_mask.view(-1, 1).expand(-1, self.num_labels).bool())
                    noised_logits = noised_logits.masked_select(
                        r1_mask.view(-1, 1).expand(-1, self.num_labels).bool())

                    r1_loss_f = KL(noised_logits.view(-1, self.num_labels),
                                   logits.view(-1, self.num_labels).detach())
                    r1_loss_b = KL(logits.view(-1, self.num_labels),
                                   noised_logits.view(-1, self.num_labels).detach())
                    r1_loss = (r1_loss_b + r1_loss_f) * self.r1_lambda
                else:
                    r1_loss = loss.data.new([0.0])

                loss = original_loss + noised_loss + r1_loss + r2_loss
                outputs = (loss, original_loss, noised_loss, r1_loss, r2_loss) + outputs[1:]

        return outputs  # (loss), logits, (hidden_states), (attentions)


def JSD_probs(x, y):
    KLDivLoss = nn.KLDivLoss(reduction='sum')
    log_mean_output = ((x + y) / 2).log()
    return (KLDivLoss(log_mean_output, x) + KLDivLoss(log_mean_output, y)) / 2


def MSE_probs(x, y):
    return F.mse_loss(x, y) * x.size(-1) * (x.size(0))


@add_start_docstrings(
    """XLM-RoBERTa Model for cross-lingual classification (stabletune). To check consistency. """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForSequenceClassificationConsistency(BertPreTrainedModel):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, noised_data_generator=None):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

        self.noise_sampler = None
        self.enable_r1_loss = False
        self.original_loss = True
        self.noised_loss = False
        self.use_hard_labels = False
        if noised_data_generator is not None:
            self.enable_r1_loss = noised_data_generator.enable_r1_loss
            self.r1_lambda = noised_data_generator.r1_lambda
            self.r2_lambda = noised_data_generator.r2_lambda
            self.original_loss = noised_data_generator.original_loss
            self.noised_loss = noised_data_generator.noised_loss
            self.use_hard_labels = noised_data_generator.use_hard_labels
            self.augment_method = None
            self.enable_random_noise = noised_data_generator.enable_random_noise

            if noised_data_generator.enable_random_noise or self.augment_method == "gn":
                self.noise_detach_embeds = noised_data_generator.noise_detach_embeds
                if noised_data_generator.noise_type in {"normal"}:
                    self.noise_sampler = torch.distributions.normal.Normal(
                        loc=0.0, scale=noised_data_generator.noise_eps
                    )
                elif noised_data_generator.noise_type == "uniform":
                    self.noise_sampler = torch.distributions.uniform.Uniform(
                        low=-noised_data_generator.noise_eps, high=noised_data_generator.noise_eps
                    )
                else:
                    raise Exception(f"unrecognized noise type {noised_data_generator.noise_type}")

    @add_start_docstrings_to_callable(XLM_ROBERTA_START_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            noised_input_ids=None,
            noised_attention_mask=None,
            noised_token_type_ids=None
    ):
        word_embeds = self.roberta.embeddings.word_embeddings(input_ids)

        outputs = self.roberta(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=word_embeds,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        if noised_input_ids is not None:
            noised_word_embeds = self.roberta.embeddings.word_embeddings(noised_input_ids)
            assert noised_attention_mask is not None
        elif self.noise_sampler is not None:
            noised_word_embeds = self.roberta.embeddings.word_embeddings(input_ids)
            noised_attention_mask = attention_mask
            noised_token_type_ids = token_type_ids
        else:
            assert False

        if self.enable_random_noise:
            noise = self.noise_sampler.sample(sample_shape=noised_word_embeds.shape).to(noised_word_embeds)
            if self.noise_detach_embeds:
                noised_word_embeds = noised_word_embeds.detach().clone() + noise
            else:
                noised_word_embeds = noised_word_embeds + noise

        noised_outputs = self.roberta(
            input_ids=None,
            attention_mask=noised_attention_mask,
            token_type_ids=noised_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=noised_word_embeds,
        )
        noised_sequence_output = noised_outputs[0]
        noised_logits = self.classifier(noised_sequence_output)

        original_probs = F.softmax(logits.view(-1, self.num_labels), dim=-1)
        noised_probs = F.softmax(noised_logits.view(-1, self.num_labels), dim=-1)

        outputs = logits.data.new(0), logits, JSD_probs(original_probs, noised_probs), MSE_probs(original_probs,
                                                                                                 noised_probs)
        return outputs


def KL_probs(input, target):
    kl_loss = target * (torch.log(target) - torch.log(input))
    zeros = torch.zeros_like(kl_loss)
    kl_loss = torch.where(torch.min(target > 0, input > 0), kl_loss, zeros)
    return kl_loss.sum()


def get_probs(logits, mask=None, attn_mask=False):
    logits = logits.float()
    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    if mask is None:
        return probs

    if attn_mask:
        return probs.masked_select(mask)

    other_probs = probs.clone().masked_fill(mask, 0)
    other_probs = other_probs.sum(dim=-1).unsqueeze(-1)

    probs = torch.cat([probs, other_probs], dim=-1)
    mask = torch.cat([mask, other_probs.gt(0)], dim=-1)
    return probs.masked_select(mask)


@add_start_docstrings(
    """XLM-RoBERTa Model for cross-lingual question answering (stabletune)""",
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForQuestionAnsweringStable(BertPreTrainedModel):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, noised_data_generator=None):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

        self.noise_sampler = None
        self.enable_r1_loss = False
        self.original_loss = True
        self.noised_loss = False
        self.use_hard_labels = False
        self.r2_lambda = 1.0
        if noised_data_generator is not None:
            self.enable_r1_loss = noised_data_generator.enable_r1_loss
            self.r1_on_boundary_only = noised_data_generator.r1_on_boundary_only
            self.r1_lambda = noised_data_generator.r1_lambda
            self.original_loss = noised_data_generator.original_loss
            self.noised_loss = noised_data_generator.noised_loss
            self.r2_lambda = noised_data_generator.r2_lambda
            self.use_hard_labels = noised_data_generator.use_hard_labels
            self.noise_detach_embeds = noised_data_generator.noise_detach_embeds
            self.augment_method = noised_data_generator.augment_method
            self.disable_translate_labels = noised_data_generator.disable_translate_labels

            if noised_data_generator.enable_random_noise or self.augment_method == "gn":
                if noised_data_generator.noise_type in {"normal"}:
                    self.noise_sampler = torch.distributions.normal.Normal(
                        loc=0.0, scale=noised_data_generator.noise_eps
                    )
                elif noised_data_generator.noise_type == "uniform":
                    self.noise_sampler = torch.distributions.uniform.Uniform(
                        low=-noised_data_generator.noise_eps, high=noised_data_generator.noise_eps
                    )
                else:
                    raise Exception(f"unrecognized noise type {noised_data_generator.noise_type}")

    @add_start_docstrings_to_callable(XLM_ROBERTA_START_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            noised_input_ids=None,
            noised_attention_mask=None,
            noised_token_type_ids=None,
            noised_r1_mask=None,
            original_r1_mask=None,
            noised_start_positions=None,
            noised_end_positions=None,
            first_stage_model_start_logits=None,
            first_stage_model_end_logits=None,
            is_augmented=None,
    ):
        word_embeds = self.roberta.embeddings.word_embeddings(input_ids)

        if is_augmented is not None:
            # ground truth data indices
            # optional when data augmentation is considered noisy
            # if first_stage_model_start_logits is not None and (self.augment_method != "mt" or self.disable_translate_labels):
            #     gt_indices = (~is_augmented.bool()).view(-1).nonzero(as_tuple=False).view(-1).tolist()
            # else:
            #     gt_indices = list(range(0, input_ids.size(0)))

            gt_indices = list(range(0, input_ids.size(0)))
            # optional to conduct r2 on original corpus
            augmented_indices = is_augmented.view(-1).nonzero(as_tuple=False).view(-1).tolist()
        else:
            gt_indices = list(range(0, input_ids.size(0)))
            augmented_indices = None

        if is_augmented is not None and self.noise_sampler is not None:
            noise = self.noise_sampler.sample(sample_shape=word_embeds.shape).to(word_embeds)
            if self.noise_detach_embeds:
                noised_word_embeds = word_embeds.detach().clone() + noise
            else:
                noised_word_embeds = word_embeds + noise
            if len(augmented_indices) > 0:
                word_embeds[augmented_indices] = noised_word_embeds[augmented_indices]

        outputs = self.roberta(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=word_embeds,
        )

        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            if is_augmented is not None:
                if len(gt_indices) > 0:
                    start_loss = loss_fct(start_logits[gt_indices], start_positions[gt_indices])
                    end_loss = loss_fct(end_logits[gt_indices], end_positions[gt_indices])
                else:
                    start_loss, end_loss = start_logits.data.new([0.0]), start_logits.data.new([0.0])
            else:
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)

            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

            if self.training:
                if first_stage_model_start_logits is not None and first_stage_model_end_logits is not None and is_augmented is not None and len(
                        augmented_indices) > 0:
                    # optional to use hard labels and augmented indices when data augmentation is considered noisy
                    if self.use_hard_labels:
                        hard_start_positions = first_stage_model_start_logits.max(dim=-1)[1]
                        hard_end_positions = first_stage_model_start_logits.max(dim=-1)[1]
                        r2_start_loss = loss_fct(start_logits[augmented_indices],
                                                 hard_start_positions[augmented_indices])
                        r2_end_loss = loss_fct(end_logits[augmented_indices], hard_end_positions[augmented_indices])
                        r2_loss = (r2_start_loss + r2_end_loss) / 2 * self.r2_lambda
                    else:
                        original_start_probs = get_probs(start_logits[augmented_indices],
                                                         attention_mask.bool()[augmented_indices], attn_mask=True)
                        original_end_probs = get_probs(end_logits[augmented_indices],
                                                       attention_mask.bool()[augmented_indices], attn_mask=True)
                        stable_start_probs = get_probs(first_stage_model_start_logits[augmented_indices],
                                                       attention_mask.bool()[augmented_indices], attn_mask=True)
                        stable_end_probs = get_probs(first_stage_model_end_logits[augmented_indices],
                                                     attention_mask.bool()[augmented_indices], attn_mask=True)

                        r2_start_loss = KL_probs(original_start_probs, stable_start_probs.detach()) / len(
                            augmented_indices)
                        r2_end_loss = KL_probs(original_end_probs, stable_end_probs.detach()) / len(augmented_indices)

                        r2_loss = (r2_start_loss + r2_end_loss) / 2 * self.r2_lambda
                else:
                    r2_loss = total_loss.data.new([0.0])

                if self.enable_r1_loss or self.noised_loss:
                    original_r1_mask = original_r1_mask.eq(1)
                    noised_r1_mask = noised_r1_mask.eq(1)
                    if noised_input_ids is not None:
                        noised_word_embeds = self.roberta.embeddings.word_embeddings(noised_input_ids)
                        assert noised_attention_mask is not None
                    elif self.noise_sampler is not None:
                        noised_word_embeds = self.roberta.embeddings.word_embeddings(input_ids)
                        noised_attention_mask = attention_mask
                        noised_token_type_ids = token_type_ids
                    else:
                        assert False

                    if self.noise_sampler is not None:
                        noise = self.noise_sampler.sample(sample_shape=noised_word_embeds.shape).to(noised_word_embeds)
                        if self.noise_detach_embeds:
                            noised_word_embeds = noised_word_embeds.detach().clone() + noise
                        else:
                            noised_word_embeds = noised_word_embeds + noise

                    noised_outputs = self.roberta(
                        input_ids=None,
                        attention_mask=noised_attention_mask,
                        token_type_ids=noised_token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=noised_word_embeds,
                    )
                    noised_sequence_output = noised_outputs[0]
                    noised_logits = self.qa_outputs(noised_sequence_output)
                    noised_start_logits, noised_end_logits = noised_logits.split(1, dim=-1)
                    noised_start_logits = noised_start_logits.squeeze(-1)
                    noised_end_logits = noised_end_logits.squeeze(-1)

                if self.original_loss:
                    original_loss = total_loss
                else:
                    original_loss = total_loss.data.new([0.0])

                if self.noised_loss:
                    ignored_index = noised_start_logits.size(1)
                    noised_start_positions.clamp_(0, ignored_index)
                    noised_end_positions.clamp_(0, ignored_index)

                    loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                    noised_start_loss = loss_fct(noised_start_logits, noised_start_positions)
                    noised_end_loss = loss_fct(noised_end_logits, noised_end_positions)
                    noised_loss = (noised_start_loss + noised_end_loss) / 2
                else:
                    noised_loss = total_loss.data.new([0.0])

                if self.enable_r1_loss:
                    if self.r1_on_boundary_only:
                        noised_start_probs = get_probs(noised_start_logits)
                        original_start_probs = get_probs(start_logits)

                        noised_end_probs = get_probs(noised_end_logits)
                        original_end_probs = get_probs(end_logits)

                        # (batch_size, seq_len) -> (batch_size, 1)
                        noised_start_probs = noised_start_probs.gather(dim=1,
                                                                       index=noised_start_positions.unsqueeze(-1))
                        noised_end_probs = noised_end_probs.gather(dim=1, index=noised_end_positions.unsqueeze(-1))

                        original_start_probs = original_start_probs.gather(dim=1,
                                                                           index=start_positions.unsqueeze(-1))
                        original_end_probs = original_end_probs.gather(dim=1, index=end_positions.unsqueeze(-1))

                        noised_start_probs = torch.cat([noised_start_probs, 1 - noised_start_probs], dim=-1)
                        noised_end_probs = torch.cat([noised_end_probs, 1 - noised_end_probs], dim=-1)

                        original_start_probs = torch.cat([original_start_probs, 1 - original_start_probs], dim=-1)
                        original_end_probs = torch.cat([original_end_probs, 1 - original_end_probs], dim=-1)

                        noised_start_probs = noised_start_probs.view(-1)
                        noised_end_probs = noised_end_probs.view(-1)
                        original_start_probs = original_start_probs.view(-1)
                        original_end_probs = original_end_probs.view(-1)
                    else:
                        noised_start_probs = get_probs(noised_start_logits, noised_r1_mask)
                        original_start_probs = get_probs(start_logits, original_r1_mask)

                        noised_end_probs = get_probs(noised_end_logits, noised_r1_mask)
                        original_end_probs = get_probs(end_logits, original_r1_mask)

                    start_r1_loss_f = KL_probs(noised_start_probs, original_start_probs.detach()) / input_ids.size(0)
                    start_r1_loss_b = KL_probs(original_start_probs, noised_start_probs.detach()) / input_ids.size(0)
                    end_r1_loss_f = KL_probs(noised_end_probs, original_end_probs.detach()) / input_ids.size(0)
                    end_r1_loss_b = KL_probs(original_end_probs, noised_end_probs.detach()) / input_ids.size(0)

                    start_r1_loss = (start_r1_loss_b + start_r1_loss_f) / 2.0
                    end_r1_loss = (end_r1_loss_b + end_r1_loss_f) / 2.0

                    r1_loss = (start_r1_loss + end_r1_loss) * self.r1_lambda
                else:
                    r1_loss = total_loss.data.new([0.0])

                loss = original_loss + noised_loss + r1_loss + r2_loss

                outputs = (loss, original_loss, noised_loss, r1_loss, r2_loss) + outputs[1:]

        return outputs  # (loss), logits, (hidden_states), (attentions)


def get_label_probs(logits, mask):
    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    # probs = F.softmax(logits, dim=-1)
    probs = probs.masked_fill(~mask.unsqueeze(-1).expand(-1, -1, probs.size(-1)), 0.0)
    n_position = torch.sum(mask.long(), dim=-1).unsqueeze(-1)
    # print(probs.size(), n_position.size())
    label_probs = torch.sum(probs, dim=1) / n_position

    # print(label_probs[0, :])
    return label_probs


def get_average_representations(output, mask):
    output = output.masked_fill(~mask.bool().unsqueeze(-1).expand(-1, -1, output.size(-1)), 0.0)
    sum_reps = torch.sum(output, dim=1)
    n_position = torch.sum(mask.long(), dim=-1).unsqueeze(-1)
    assert torch.min(n_position.view(-1)) > 0

    ave_reps = sum_reps / n_position
    return ave_reps


def get_align_probs(logits, pooling_ids):
    # (bsz, seq_len)
    pooling_count = torch.zeros_like(pooling_ids)
    pooling_count.scatter_add_(dim=1, index=pooling_ids,
                               src=torch.ones_like(pooling_ids))
    mask = pooling_count.ne(0)
    mask[:, 0] = 0

    # (bsz, seq_len, num_labels)
    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    sum_probs = torch.zeros_like(probs)
    expanded_pooling_ids = pooling_ids.unsqueeze(-1).expand(-1, -1, probs.size(-1))
    sum_probs.scatter_add_(dim=1, index=expanded_pooling_ids, src=probs)

    # avoid from dividing zero
    pooling_count.masked_fill_(pooling_count.eq(0), 1.0)
    sum_probs = sum_probs.div(pooling_count.unsqueeze(-1).expand(-1, -1, probs.size(-1)))

    return pooling_count, sum_probs, mask


@add_start_docstrings(
    """XLM-RoBERTa Model transformer for cross-lingual sequential labeling (stabletune). 
    Optional with sum-pooling strategy""",
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForTokenClassificationPoolingStable(BertPreTrainedModel):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, noised_data_generator=None, use_pooling_strategy=False):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

        self.use_pooling_strategy = use_pooling_strategy
        self.noise_sampler = None
        self.enable_r1_loss = False
        self.original_loss = True
        self.noised_loss = False
        self.detach_embeds = False
        if noised_data_generator is not None:
            self.enable_r1_loss = noised_data_generator.enable_r1_loss
            self.r1_lambda = noised_data_generator.r1_lambda
            self.original_loss = noised_data_generator.original_loss
            self.noised_loss = noised_data_generator.noised_loss
            self.use_sentence_label_probs = noised_data_generator.use_sentence_label_probs
            self.use_token_label_probs = noised_data_generator.use_token_label_probs
            self.use_align_label_probs = noised_data_generator.use_align_label_probs
            self.r2_lambda = noised_data_generator.r2_lambda
            self.use_average_representations = noised_data_generator.use_average_representations
            self.detach_embeds = noised_data_generator.detach_embeds
            self.disable_backward_kl = noised_data_generator.disable_backward_kl
            self.use_hard_labels = noised_data_generator.use_hard_labels
            self.augment_method = noised_data_generator.augment_method

            if not (noised_data_generator.original_loss or noised_data_generator.enable_r1_loss):
                # replace original dataset to noised dataset
                assert self.noised_loss
                self.noised_loss = False
                self.original_loss = True

            if noised_data_generator.enable_random_noise:
                if noised_data_generator.noise_type in {"normal"}:
                    self.noise_sampler = torch.distributions.normal.Normal(
                        loc=0.0, scale=noised_data_generator.noise_eps
                    )
                elif noised_data_generator.noise_type == "uniform":
                    self.noise_sampler = torch.distributions.uniform.Uniform(
                        low=-noised_data_generator.noise_eps, high=noised_data_generator.noise_eps
                    )
                else:
                    raise Exception(f"unrecognized noise type {noised_data_generator.noise_type}")

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            pooling_ids=None,
            noised_input_ids=None,
            noised_attention_mask=None,
            noised_token_type_ids=None,
            noised_labels=None,
            noised_pooling_ids=None,
            noised_r1_mask=None,
            original_r1_mask=None,
            src_pooling_ids=None,
            tgt_pooling_ids=None,
            is_augmented=None,
            first_stage_model_logits=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForTokenClassification
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

        """

        word_embeds = self.roberta.embeddings.word_embeddings(input_ids)

        if is_augmented is not None:
            # ground truth data indices
            # optional when data augmentation is considered noisy
            # if self.augment_method != "mt" and first_stage_model_logits is not None:
            #     gt_indices = (~is_augmented.bool()).view(-1).nonzero(as_tuple=False).view(-1).tolist()
            # else:
            #     gt_indices = list(range(0, input_ids.size(0)))

            gt_indices = list(range(0, input_ids.size(0)))
            # optional to conduct r2 on original corpus
            augmented_indices = is_augmented.view(-1).nonzero(as_tuple=False).view(-1).tolist()
        else:
            gt_indices = list(range(0, input_ids.size(0)))
            augmented_indices = None

        if is_augmented is not None and self.noise_sampler is not None:
            noise = self.noise_sampler.sample(sample_shape=word_embeds.shape).to(word_embeds)
            if self.detach_embeds:
                noised_word_embeds = word_embeds.detach().clone() + noise
            else:
                noised_word_embeds = word_embeds + noise
            if len(augmented_indices) > 0:
                # print(word_embeds[indices].size(), indices, noised_word_embeds[indices].size())
                word_embeds[augmented_indices] = noised_word_embeds[augmented_indices]

        outputs = self.roberta(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=word_embeds.detach().clone() if self.detach_embeds else word_embeds,
        )

        # (batch_size, seq_len, hidden_size)
        sequence_output = outputs[0]

        if self.use_pooling_strategy:
            sum_sequence_output = torch.zeros_like(sequence_output)
            expanded_pooling_ids = pooling_ids.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1))
            sum_sequence_output.scatter_add_(dim=1, index=expanded_pooling_ids, src=sequence_output)

            pooled_sequence_output = self.dropout(sum_sequence_output)
            logits = self.classifier(pooled_sequence_output)
        else:
            sequence_output = self.dropout(sequence_output)
            logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask[gt_indices].view(-1) == 1
                active_logits = logits[gt_indices].view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels[gt_indices].view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits[gt_indices].view(-1, self.num_labels), labels[gt_indices].view(-1))
            outputs = (loss,) + outputs

            if self.training:
                if first_stage_model_logits is not None and is_augmented is not None and len(augmented_indices) > 0:
                    sum_token_count = torch.zeros_like(pooling_ids)
                    sum_token_count.scatter_add_(dim=1, index=pooling_ids, src=torch.ones_like(pooling_ids))
                    sum_token_count[:, 0] = 0
                    # optional to use hard labels and augmented indices when data augmenation is considered noisy
                    # hard labels for NER task
                    if self.use_hard_labels:
                        hard_labels = first_stage_model_logits.max(dim=-1)[1]
                        active_loss = sum_token_count[augmented_indices].gt(0).view(-1) == 1
                        active_logits = logits[augmented_indices].view(-1, self.num_labels)
                        active_labels = torch.where(active_loss, hard_labels[augmented_indices].view(-1),
                                                    torch.tensor(loss_fct.ignore_index).type_as(labels))
                        r2_loss = loss_fct(active_logits, active_labels) * self.r2_lambda
                    else:
                        sum_mask = sum_token_count[augmented_indices].view(-1).gt(0).unsqueeze(-1).expand(-1,
                                                                                                          self.num_labels)
                        token_teacher_logits = first_stage_model_logits[augmented_indices].view(-1,
                                                                                                self.num_labels).masked_select(
                            sum_mask).view(-1, self.num_labels)
                        token_student_logits = logits[augmented_indices].view(-1, self.num_labels).masked_select(
                            sum_mask).view(-1, self.num_labels)
                        r2_loss = KL(token_student_logits, token_teacher_logits.detach()) * self.r2_lambda
                else:
                    r2_loss = loss.data.new([0.0])

                if self.enable_r1_loss or self.noised_loss:
                    if noised_input_ids is not None:
                        noised_word_embeds = self.roberta.embeddings.word_embeddings(noised_input_ids)
                        assert noised_attention_mask is not None
                    elif self.noise_sampler is not None:
                        noised_word_embeds = self.roberta.embeddings.word_embeddings(input_ids)
                        noised_attention_mask = attention_mask
                        noised_token_type_ids = token_type_ids
                    else:
                        assert False

                    if self.noise_sampler is not None:
                        noise = self.noise_sampler.sample(sample_shape=noised_word_embeds.shape).to(noised_word_embeds)
                        if self.detach_embeds:
                            noised_word_embeds = noised_word_embeds.detach().clone() + noise
                        else:
                            noised_word_embeds = noised_word_embeds + noise
                    else:
                        noised_word_embeds = noised_word_embeds.detach().clone() if self.detach_embeds else noised_word_embeds

                    noised_outputs = self.roberta(
                        input_ids=None,
                        attention_mask=noised_attention_mask,
                        token_type_ids=noised_token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=noised_word_embeds,
                    )

                    noised_sequence_output = noised_outputs[0]
                    if self.use_pooling_strategy:
                        noised_sum_token_count = torch.zeros_like(noised_pooling_ids)
                        noised_sum_token_count.scatter_add_(dim=1, index=noised_pooling_ids,
                                                            src=torch.ones_like(noised_pooling_ids))
                        noised_sum_sequence_output = torch.zeros_like(noised_sequence_output)
                        noised_expanded_pooling_ids = noised_pooling_ids.unsqueeze(-1).expand(-1, -1,
                                                                                              noised_sequence_output.size(
                                                                                                  -1))
                        noised_sum_sequence_output.scatter_add_(dim=1, index=noised_expanded_pooling_ids,
                                                                src=noised_sequence_output)

                        pooled_noised_sequence_output = self.dropout(noised_sum_sequence_output)
                        noised_logits = self.classifier(pooled_noised_sequence_output)
                    else:
                        noised_sequence_output = self.dropout(noised_sequence_output)
                        noised_logits = self.classifier(noised_sequence_output)

                if self.original_loss:
                    original_loss = loss
                else:
                    original_loss = loss.data.new([0.0])

                if self.noised_loss:
                    if noised_attention_mask is not None:
                        noised_active_loss = noised_attention_mask.view(-1) == 1
                        noised_active_logits = noised_logits.view(-1, self.num_labels)
                        noised_active_labels = torch.where(
                            noised_active_loss, noised_labels.view(-1),
                            torch.tensor(loss_fct.ignore_index).type_as(noised_labels)
                        )
                        noised_loss = loss_fct(noised_active_logits, noised_active_labels)
                    else:
                        noised_loss = loss_fct(noised_logits.view(-1, self.num_labels), noised_labels.view(-1))
                else:
                    noised_loss = loss.data.new([0.0])

                if self.enable_r1_loss:

                    if self.use_align_label_probs:
                        src_pooling_count, src_probs, src_mask = get_align_probs(logits, src_pooling_ids)
                        tgt_pooling_count, tgt_probs, tgt_mask = get_align_probs(noised_logits, tgt_pooling_ids)

                        assert src_mask.eq(tgt_mask).sum() == src_mask.size(0) * src_mask.size(1)

                        indices = src_mask.view(-1).nonzero(as_tuple=False).view(-1).tolist()

                        src_probs = src_probs.view(-1, src_probs.size(-1))[indices]
                        tgt_probs = tgt_probs.view(-1, src_probs.size(-1))[indices]

                        align_r1_loss_f = KL_probs(src_probs, tgt_probs.detach()) / src_probs.size(0)
                        align_r1_loss_b = KL_probs(tgt_probs, src_probs.detach()) / src_probs.size(0)
                        align_r1_loss = align_r1_loss_b + align_r1_loss_f
                    else:
                        align_r1_loss = loss.data.new([0.0])

                    if self.use_token_label_probs:
                        original_indices = original_r1_mask.view(-1).eq(1).nonzero(as_tuple=False).view(-1).tolist()
                        noised_indices = noised_r1_mask.view(-1).eq(1).nonzero(as_tuple=False).view(-1).tolist()
                        token_original_logits = logits.view(-1, self.num_labels)[original_indices]
                        token_noised_logits = noised_logits.view(-1, self.num_labels)[noised_indices]

                        token_r1_loss_f = KL(token_noised_logits, token_original_logits.detach())
                        token_r1_loss_b = KL(token_original_logits, token_noised_logits.detach())
                        if not self.disable_backward_kl:
                            token_r1_loss = token_r1_loss_f + token_r1_loss_b
                        else:
                            token_r1_loss = token_r1_loss_f
                    else:
                        token_r1_loss = loss.data.new([0.0])

                    if self.use_sentence_label_probs:
                        original_probs = get_label_probs(logits, labels.ne(loss_fct.ignore_index))
                        noised_probs = get_label_probs(noised_logits, noised_labels.ne(loss_fct.ignore_index))

                        sentence_r1_loss_f = KL_probs(noised_probs, original_probs.detach()) / input_ids.size(0)
                        sentence_r1_loss_b = KL_probs(original_probs, noised_probs.detach()) / input_ids.size(0)
                        sentence_r1_loss = sentence_r1_loss_f + sentence_r1_loss_b
                    else:
                        sentence_r1_loss = loss.data.new([0.0])

                    r1_loss = (token_r1_loss + sentence_r1_loss + align_r1_loss) * self.r1_lambda
                else:
                    r1_loss = loss.data.new([0.0])

                loss = original_loss + noised_loss + r1_loss + r2_loss

                # print(loss, original_loss, r1_loss, loss.eq(original_loss), loss - r1_loss)
                outputs = (loss, original_loss, noised_loss, r1_loss, r2_loss) + outputs[1:]

        return outputs  # (loss), scores, (hidden_states), (attentions)
