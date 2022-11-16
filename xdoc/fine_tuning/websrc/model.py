import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertModel, BertPreTrainedModel, RobertaConfig
# from transformers.modeling_bert import BertLayerNorm, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
BertLayerNorm = torch.nn.LayerNorm
logger = logging.getLogger(__name__)

LAYOUTLMV1_PRETRAINED_MODEL_ARCHIVE_MAP = {}

LAYOUTLMV1_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class Layoutlmv1Config_roberta(RobertaConfig):
    pretrained_config_archive_map = LAYOUTLMV1_PRETRAINED_CONFIG_ARCHIVE_MAP 
    model_type = "bert"

    def __init__(self, max_2d_position_embeddings=1024, add_linear=False, **kwargs):
        super().__init__(**kwargs)
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.add_linear = add_linear # determine whether to add an additional mapping



class Layoutlmv1Config(BertConfig):
    pretrained_config_archive_map = LAYOUTLMV1_PRETRAINED_CONFIG_ARCHIVE_MAP 
    model_type = "bert"

    def __init__(self, max_2d_position_embeddings=1024, add_linear=False, **kwargs):
        super().__init__(**kwargs)
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.add_linear = add_linear # determine whether to add an additional mapping



class WebConfig:
    max_depth = 50
    xpath_unit_hidden_size = 32
    hidden_size = 768
    hidden_dropout_prob = 0.1
    layer_norm_eps = 1e-12
    max_xpath_tag_unit_embeddings = 256
    max_xpath_subs_unit_embeddings = 1024




class XPathEmbeddings(nn.Module):
    """Construct the embddings from xpath -- tag and subscript"""

    # we drop tree-id in this version, as its info can be covered by xpath

    def __init__(self, config):
        super(XPathEmbeddings, self).__init__()
        config = WebConfig()
        self.max_depth = config.max_depth

        self.xpath_unitseq2_embeddings = nn.Linear(
            config.xpath_unit_hidden_size * self.max_depth, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.activation = nn.ReLU()
        self.xpath_unitseq2_inner = nn.Linear(config.xpath_unit_hidden_size * self.max_depth, 4 * config.hidden_size)
        self.inner2emb = nn.Linear(4 * config.hidden_size, config.hidden_size)

        self.xpath_tag_sub_embeddings = nn.ModuleList(
            [nn.Embedding(config.max_xpath_tag_unit_embeddings, config.xpath_unit_hidden_size) for _ in
             range(self.max_depth)])

        self.xpath_subs_sub_embeddings = nn.ModuleList(
            [nn.Embedding(config.max_xpath_subs_unit_embeddings, config.xpath_unit_hidden_size) for _ in
             range(self.max_depth)])

    def forward(self,
                xpath_tags_seq=None,
                xpath_subs_seq=None):
        xpath_tags_embeddings = []
        xpath_subs_embeddings = []

        for i in range(self.max_depth):
            xpath_tags_embeddings.append(self.xpath_tag_sub_embeddings[i](xpath_tags_seq[:, :, i]))
            xpath_subs_embeddings.append(self.xpath_subs_sub_embeddings[i](xpath_subs_seq[:, :, i]))

        xpath_tags_embeddings = torch.cat(xpath_tags_embeddings, dim=-1)
        xpath_subs_embeddings = torch.cat(xpath_subs_embeddings, dim=-1)

        xpath_embeddings = xpath_tags_embeddings + xpath_subs_embeddings

        xpath_embeddings = self.inner2emb(
            self.dropout(self.activation(self.xpath_unitseq2_inner(xpath_embeddings))))

        return xpath_embeddings


class Layoutlmv1Embeddings(nn.Module):
    def __init__(self, config):
        super(Layoutlmv1Embeddings, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.x_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.y_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.h_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.w_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # for web extension
        self.xpath_embeddings = XPathEmbeddings(config)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.doc_linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.doc_linear2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.web_linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.web_linear2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.web_linear3 = nn.Linear(config.hidden_size, config.hidden_size)
        self.web_linear4 = nn.Linear(config.hidden_size, config.hidden_size)

        self.relu = nn.ReLU()

    def forward(
        self,
        input_ids,
        bbox=None,
        xpath_tags_seq=None,
        xpath_subs_seq=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        embedding_mode=None
    ):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if embedding_mode != None and embedding_mode == 'box' : # doc entry
           
            bbox = torch.clamp(bbox, 0, self.config.max_2d_position_embeddings-1)
            
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
           
            embeddings = (
                words_embeddings
                + position_embeddings
                + left_position_embeddings
                + upper_position_embeddings
                # + right_position_embeddings
                # + lower_position_embeddings
                # + h_position_embeddings
                # + w_position_embeddings
                + token_type_embeddings
            )
        elif embedding_mode != None and embedding_mode == 'html+box' : # doc entry
          
            bbox = torch.clamp(bbox, 0, self.config.max_2d_position_embeddings-1)
            
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            xpath_embeddings = self.xpath_embeddings(xpath_tags_seq, xpath_subs_seq)

            embeddings = (
                words_embeddings
                + position_embeddings
                + left_position_embeddings
                + upper_position_embeddings
                + xpath_embeddings
                # + right_position_embeddings
                # + lower_position_embeddings
                # + h_position_embeddings
                # + w_position_embeddings
                + token_type_embeddings
            )
        else: # web entry
            if not self.config.add_linear:
                xpath_embeddings = self.xpath_embeddings(xpath_tags_seq, xpath_subs_seq)
                embeddings = (
                    words_embeddings
                    + position_embeddings
                    + token_type_embeddings
                    + xpath_embeddings
                )
            else:
                xpath_embeddings = self.xpath_embeddings(xpath_tags_seq, xpath_subs_seq)

                temp_embeddings = self.web_linear2(self.relu(self.web_linear1(
                    xpath_embeddings
                )))
                embeddings = (
                    words_embeddings
                    + position_embeddings
                    + token_type_embeddings
                    + temp_embeddings
                )

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Layoutlmv1Model(BertModel):

    config_class = Layoutlmv1Config
    pretrained_model_archive_map = LAYOUTLMV1_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super(Layoutlmv1Model, self).__init__(config)
        self.embeddings = Layoutlmv1Embeddings(config)
        self.init_weights()

    def forward(
        self,
        input_ids,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        xpath_tags_seq=None,
        xpath_subs_seq=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        embedding_mode=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) 
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32
            # dtype=next(self.parameters()).dtype # this will trigger error when using high version torch
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None: 
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, bbox=bbox, xpath_tags_seq=xpath_tags_seq, xpath_subs_seq=xpath_subs_seq, position_ids=position_ids, token_type_ids=token_type_ids, embedding_mode=embedding_mode
        )
        encoder_outputs = self.encoder(
            embedding_output, extended_attention_mask, head_mask=head_mask
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class Layoutlmv1ForTokenClassification(BertPreTrainedModel):
    config_class = Layoutlmv1Config
    pretrained_model_archive_map = LAYOUTLMV1_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = Layoutlmv1Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[
            2:
        ]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class Layoutlmv1ForMaskedLM(BertPreTrainedModel):
    config_class = Layoutlmv1Config
    pretrained_model_archive_map = LAYOUTLMV1_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)

        self.bert = Layoutlmv1Model(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
        xpath_tags_seq=None,
        xpath_subs_seq=None,
    ):

        outputs = self.bert(
            input_ids,
            bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[
            2:
        ]  # Add hidden states and attention if they are here

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
            )
            outputs = (masked_lm_loss,) + outputs
        return (
            outputs
        )  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)


class Layoutlmv1ForMaskedLM_roberta(BertPreTrainedModel):
    config_class = Layoutlmv1Config
    pretrained_model_archive_map = LAYOUTLMV1_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = Layoutlmv1Model(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.roberta.embeddings.word_embeddings

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
        xpath_tags_seq=None,
        xpath_subs_seq=None,
    ):

        outputs = self.roberta(
            input_ids,
            bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[
            2:
        ]  # Add hidden states and attention if they are here

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
            )
            outputs = (masked_lm_loss,) + outputs
        return (
            outputs
        )  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)



class Layoutlmv1ForQuestionAnswering(BertPreTrainedModel):
    config_class = Layoutlmv1Config
    pretrained_model_archive_map = LAYOUTLMV1_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = Layoutlmv1Model(config) 
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        # inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        # output_attentions=None,
        # output_hidden_states=None,
        # return_dict=None,
        xpath_tags_seq=None,
        xpath_subs_seq=None,
        embedding_mode=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids,
            bbox=bbox,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            embedding_mode=embedding_mode
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
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

        # if not return_dict:
        #     output = (start_logits, end_logits) + outputs[2:]
        #     return ((total_loss,) + output) if total_loss is not None else output
        #
        # return QuestionAnsweringModelOutput(
        #     loss=total_loss,
        #     start_logits=start_logits,
        #     end_logits=end_logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

        output = (start_logits, end_logits) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output



class Layoutlmv1ForQuestionAnswering_roberta(BertPreTrainedModel):
    config_class = Layoutlmv1Config
    pretrained_model_archive_map = LAYOUTLMV1_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = Layoutlmv1Model(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        # inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        # output_attentions=None,
        # output_hidden_states=None,
        # return_dict=None,
        xpath_tags_seq=None,
        xpath_subs_seq=None,
        embedding_mode=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids=input_ids,
            bbox=bbox,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            embedding_mode=embedding_mode
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
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

        # if not return_dict:
        #     output = (start_logits, end_logits) + outputs[2:]
        #     return ((total_loss,) + output) if total_loss is not None else output
        #
        # return QuestionAnsweringModelOutput(
        #     loss=total_loss,
        #     start_logits=start_logits,
        #     end_logits=end_logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

        output = (start_logits, end_logits) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output
