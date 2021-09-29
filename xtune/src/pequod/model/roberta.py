import torch

from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_bert import BertPreTrainedModel, BertForQuestionAnswering
from transformers.modeling_roberta import RobertaModel


class RobertaForQuestionAnswering(BertPreTrainedModel):

  base_model_prefix = "roberta"
  def __init__(self, config):
    BertPreTrainedModel.__init__(self, config)
    self.num_labels = config.num_labels
    self.roberta = RobertaModel(config)
    self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
    BertPreTrainedModel.init_weights(self)
  
  def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, start_positions=None, end_positions=None, **kwargs):

    outputs = self.roberta(input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids, 
      head_mask=head_mask,
      **kwargs)

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
      start_loss = loss_fct(start_logits, start_positions)
      end_loss = loss_fct(end_logits, end_positions)
      total_loss = (start_loss + end_loss) / 2
      outputs = (total_loss,) + outputs

    return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)