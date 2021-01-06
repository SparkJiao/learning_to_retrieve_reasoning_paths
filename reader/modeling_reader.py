# from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from torch import nn
from torch.nn import CrossEntropyLoss
import torch

from transformers.modeling_bert import BertModel, BertPreTrainedModel
from transformers.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaConfig

from modules.bert_iter_models import IterBertPreTrainedConfig, IterBertModel
from modules.layers import Pooler, weighted_sum


class BERTLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class BertForQuestionAnsweringConfidence(BertPreTrainedModel):

    def __init__(self, config, no_masking, lambda_scale=1.0):
        super(BertForQuestionAnsweringConfidence, self).__init__(config)
        self.bert = BertModel(config)
        # self.num_labels = num_labels
        self.num_labels = config.num_labels
        self.no_masking = no_masking
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(
            config.hidden_size, 2)  # [N, L, H] => [N, L, 2]
        self.qa_classifier = nn.Linear(
            config.hidden_size, self.num_labels)  # [N, H] => [N, n_class]
        self.lambda_scale = lambda_scale

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(
                    mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(
                    mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(
                    mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()

        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask,
                start_positions=None, end_positions=None, switch_list=None):
        sequence_output, pooled_output = self.bert(input_ids=input_ids,
                                                   token_type_ids=token_type_ids,
                                                   attention_mask=attention_mask)

        # Calculate the sequence logits.
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # Calculate the class logits
        pooled_output = self.dropout(pooled_output)
        switch_logits = self.qa_classifier(pooled_output)

        if start_positions is not None and end_positions is not None and switch_list is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(
                ignore_index=ignored_index, reduction='none')

            # if no_masking is True, we do not mask the no-answer examples'
            # span losses.
            if self.no_masking is True:
                start_losses = loss_fct(start_logits, start_positions)
                end_losses = loss_fct(end_logits, end_positions)

            else:
                # You care about the span only when switch is 0
                span_mask = (switch_list == 0).type(torch.FloatTensor).cuda()
                start_losses = loss_fct(
                    start_logits, start_positions) * span_mask
                end_losses = loss_fct(end_logits, end_positions) * span_mask

            switch_losses = loss_fct(switch_logits, switch_list)
            assert len(start_losses) == len(
                end_losses) == len(switch_losses)
            return (self.lambda_scale * (start_losses + end_losses) + switch_losses).mean()

        elif start_positions is None or end_positions is None or switch_list is None:
            return start_logits, end_logits, switch_logits

        else:
            raise NotImplementedError()


class IterBertForQuestionAnsweringConfidence(IterBertModel):

    def __init__(self, config: IterBertPreTrainedConfig, no_masking, lambda_scale=1.0):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.no_masking = no_masking
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.retrieval_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.retrieval_k = nn.Linear(config.hidden_size, config.hidden_size)

        self.retrieval_o = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Tanh()
        )

        self.qa_outputs = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        # self.qa_outputs = nn.Linear(config.hidden_size, 2)  # [N, L, H] => [N, L, 2]
        self.qa_classifier = nn.Linear(config.hidden_size, self.num_labels)  # [N, H] => [N, n_class]
        self.lambda_scale = lambda_scale

        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask,
                start_positions=None, end_positions=None, switch_list=None):

        batch, seq_len = input_ids.size()

        # `token_type_ids`: [0,0,0,1,1,1,1,0,0,0]
        # `attention_mask`: [1,1,1,1,1,1,1,0,0,0]
        # `1` for true token and `0` for mask
        question_mask = (1 - token_type_ids) * attention_mask
        passage_mask = token_type_ids * attention_mask

        seq_output = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)[0]

        cls_h = seq_output[:, :1]
        question_mask = question_mask.to(seq_output.dtype)
        passage_mask = passage_mask.to(seq_output.dtype)
        attention_mask = attention_mask.to(seq_output.dtype)

        q_hidden = self.query(cls_h, seq_output.unsqueeze(1), 1 - question_mask.unsqueeze(1),
                              aligned=True, residual=False).view(batch, seq_output.size(-1))

        retrieve_q = self.retrieval_q(q_hidden)
        retrieve_k = self.retrieval_k(seq_output)
        retrieve_s = torch.einsum("bh,bsh->bs", retrieve_q, retrieve_k)
        retrieve_a = torch.softmax(retrieve_s + (1 - passage_mask) * -10000.0, dim=1)
        # retrieve_o = self.dropout(self.retrieval_o(torch.cat([
        #     q_hidden, torch.einsum("bs,bsh->bh", retrieve_a, seq_output)
        # ], dim=-1)))
        # FIX at 2021/1/4 remove the duplicate dropout
        retrieve_o = self.retrieval_o(torch.cat([
            q_hidden, torch.einsum("bs,bsh->bh", retrieve_a, seq_output)], dim=-1))

        # Calculate the sequence logits
        bilinear_q = self.qa_outputs(q_hidden).view(batch, 2, q_hidden.size(-1))
        logits = torch.einsum("bih,bjh->bji", bilinear_q, seq_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # Calculate the class logits
        pooled_output = self.dropout(retrieve_o)
        switch_logits = self.qa_classifier(pooled_output)

        if start_positions is not None and end_positions is not None and switch_list is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduction='none')

            # if no_masking is True, we do not mask the no-answer examples'
            # span losses.
            if self.no_masking is True:
                start_losses = loss_fct(start_logits, start_positions)
                end_losses = loss_fct(end_logits, end_positions)

            else:
                # You care about the span only when switch is 0
                span_mask = (switch_list == 0).type(torch.FloatTensor).cuda()
                start_losses = loss_fct(start_logits, start_positions) * span_mask
                end_losses = loss_fct(end_logits, end_positions) * span_mask

            switch_losses = loss_fct(switch_logits, switch_list)
            assert len(start_losses) == len(end_losses) == len(switch_losses)

            return (self.lambda_scale * (start_losses + end_losses) + switch_losses).mean()

        elif start_positions is None or end_positions is None or switch_list is None:
            return start_logits, end_logits, switch_logits

        else:
            raise NotImplementedError()


class IterBertForQuestionAnsweringConfidenceV2(IterBertModel):

    def __init__(self, config: IterBertPreTrainedConfig, no_masking, lambda_scale=1.0):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.no_masking = no_masking
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.retrieval_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.retrieval_k = nn.Linear(config.hidden_size, config.hidden_size)

        self.retrieval_o = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Tanh()
        )

        self.qa_outputs = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        # self.qa_outputs = nn.Linear(config.hidden_size, 2)  # [N, L, H] => [N, L, 2]
        self.qa_classifier = nn.Linear(config.hidden_size, self.num_labels)  # [N, H] => [N, n_class]
        self.lambda_scale = lambda_scale

        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask,
                start_positions=None, end_positions=None, switch_list=None):

        batch, seq_len = input_ids.size()

        # `token_type_ids`: [0,0,0,1,1,1,1,0,0,0]
        # `attention_mask`: [1,1,1,1,1,1,1,0,0,0]
        # `1` for true token and `0` for mask
        question_mask = (1 - token_type_ids) * attention_mask
        passage_mask = token_type_ids * attention_mask

        seq_output = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)[0]

        cls_h = seq_output[:, :1]
        question_mask = question_mask.to(seq_output.dtype)
        passage_mask = passage_mask.to(seq_output.dtype)
        attention_mask = attention_mask.to(seq_output.dtype)

        q_hidden = self.query(cls_h, seq_output.unsqueeze(1), 1 - question_mask.unsqueeze(1),
                              aligned=True, residual=False).view(batch, seq_output.size(-1))

        retrieve_q = self.retrieval_q(q_hidden)
        retrieve_k = self.retrieval_k(seq_output)
        retrieve_s = torch.einsum("bh,bsh->bs", retrieve_q, retrieve_k)
        retrieve_a = torch.softmax(retrieve_s + (1 - passage_mask) * -10000.0, dim=1)
        retrieve_o = self.retrieval_o(torch.cat([
            q_hidden, torch.einsum("bs,bsh->bh", retrieve_a, seq_output)
        ], dim=-1))

        # Calculate the sequence logits
        bilinear_q = self.qa_outputs(retrieve_o).view(batch, 2, q_hidden.size(-1))
        logits = torch.einsum("bih,bjh->bji", bilinear_q, seq_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # Calculate the class logits
        pooled_output = self.dropout(retrieve_o)
        switch_logits = self.qa_classifier(pooled_output)

        if start_positions is not None and end_positions is not None and switch_list is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduction='none')

            # if no_masking is True, we do not mask the no-answer examples'
            # span losses.
            if self.no_masking is True:
                start_losses = loss_fct(start_logits, start_positions)
                end_losses = loss_fct(end_logits, end_positions)

            else:
                # You care about the span only when switch is 0
                span_mask = (switch_list == 0).type(torch.FloatTensor).cuda()
                start_losses = loss_fct(start_logits, start_positions) * span_mask
                end_losses = loss_fct(end_logits, end_positions) * span_mask

            switch_losses = loss_fct(switch_logits, switch_list)
            assert len(start_losses) == len(end_losses) == len(switch_losses)

            return (self.lambda_scale * (start_losses + end_losses) + switch_losses).mean()

        elif start_positions is None or end_positions is None or switch_list is None:
            return start_logits, end_logits, switch_logits

        else:
            raise NotImplementedError()


class IterBertForQuestionAnsweringConfidenceV3(IterBertModel):

    def __init__(self, config: IterBertPreTrainedConfig, no_masking, lambda_scale=1.0):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.no_masking = no_masking
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.retrieval_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.retrieval_k = nn.Linear(config.hidden_size, config.hidden_size)

        # self.retrieval_o = nn.Sequential(
        #     nn.Linear(config.hidden_size * 2, config.hidden_size),
        #     nn.Tanh()
        # )
        self.retrieval_o = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.qa_outputs = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        # self.qa_outputs = nn.Linear(config.hidden_size, 2)  # [N, L, H] => [N, L, 2]

        self.qa_pooler = Pooler(config.hidden_size)
        self.qa_classifier = nn.Linear(config.hidden_size, self.num_labels)  # [N, H] => [N, n_class]
        self.lambda_scale = lambda_scale

        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask,
                start_positions=None, end_positions=None, switch_list=None):

        batch, seq_len = input_ids.size()

        # `token_type_ids`: [0,0,0,1,1,1,1,0,0,0]
        # `attention_mask`: [1,1,1,1,1,1,1,0,0,0]
        # `1` for true token and `0` for mask
        question_mask = (1 - token_type_ids) * attention_mask
        passage_mask = token_type_ids * attention_mask

        seq_output = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)[0]

        cls_h = seq_output[:, :1]
        question_mask = question_mask.to(seq_output.dtype)
        passage_mask = passage_mask.to(seq_output.dtype)
        attention_mask = attention_mask.to(seq_output.dtype)

        q_hidden = self.query(cls_h, seq_output.unsqueeze(1), 1 - question_mask.unsqueeze(1),
                              aligned=True, residual=False).view(batch, seq_output.size(-1))

        retrieve_q = self.retrieval_q(q_hidden)
        retrieve_k = self.retrieval_k(seq_output)
        retrieve_s = torch.einsum("bh,bsh->bs", retrieve_q, retrieve_k)
        retrieve_a = torch.softmax(retrieve_s + (1 - passage_mask) * -10000.0, dim=1)
        retrieve_o = self.retrieval_o(torch.cat([
            q_hidden, torch.einsum("bs,bsh->bh", retrieve_a, seq_output)
        ], dim=-1))

        # Calculate the sequence logits
        bilinear_q = self.qa_outputs(retrieve_o).view(batch, 2, q_hidden.size(-1))
        logits = torch.einsum("bih,bjh->bji", bilinear_q, seq_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # Calculate the class logits
        pooled_output = self.dropout(self.qa_pooler(retrieve_o))
        switch_logits = self.qa_classifier(pooled_output)

        if start_positions is not None and end_positions is not None and switch_list is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduction='none')

            # if no_masking is True, we do not mask the no-answer examples'
            # span losses.
            if self.no_masking is True:
                start_losses = loss_fct(start_logits, start_positions)
                end_losses = loss_fct(end_logits, end_positions)

            else:
                # You care about the span only when switch is 0
                span_mask = (switch_list == 0).type(torch.FloatTensor).cuda()
                start_losses = loss_fct(start_logits, start_positions) * span_mask
                end_losses = loss_fct(end_logits, end_positions) * span_mask

            switch_losses = loss_fct(switch_logits, switch_list)
            assert len(start_losses) == len(end_losses) == len(switch_losses)

            return (self.lambda_scale * (start_losses + end_losses) + switch_losses).mean()

        elif start_positions is None or end_positions is None or switch_list is None:
            return start_logits, end_logits, switch_logits

        else:
            raise NotImplementedError()


class IterBertForQuestionAnsweringConfidenceV4(IterBertModel):

    def __init__(self, config: IterBertPreTrainedConfig, no_masking, lambda_scale=1.0):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.no_masking = no_masking
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.retrieval_sum = nn.Linear(config.hidden_size, config.hidden_size)
        self.retrieval_o = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.pooler = Pooler(config.hidden_size)

        self.qa_outputs = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        self.qa_classifier = nn.Linear(config.hidden_size, self.num_labels)  # [N, H] => [N, n_class]
        self.lambda_scale = lambda_scale

        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask,
                start_positions=None, end_positions=None, switch_list=None):

        batch, seq_len = input_ids.size()

        # `token_type_ids`: [0,0,0,1,1,1,1,0,0,0]
        # `attention_mask`: [1,1,1,1,1,1,1,0,0,0]
        # `1` for true token and `0` for mask
        question_mask = (1 - token_type_ids) * attention_mask
        passage_mask = token_type_ids * attention_mask

        seq_output = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)[0]

        cls_h = seq_output[:, :1]
        question_mask = question_mask.to(seq_output.dtype)
        passage_mask = passage_mask.to(seq_output.dtype)
        attention_mask = attention_mask.to(seq_output.dtype)

        q_hidden = self.query(cls_h, seq_output.unsqueeze(1), 1 - question_mask.unsqueeze(1),
                              aligned=True, residual=False).view(batch, seq_output.size(-1))

        retrieve_q = self.retrieval_sum(q_hidden)

        p_hidden, _ = weighted_sum(retrieve_q, seq_output, 1 - passage_mask)

        hidden = self.retrieval_o(torch.cat([q_hidden, p_hidden], dim=-1))

        # Calculate the sequence logits
        bilinear_q = self.qa_outputs(q_hidden).view(batch, 2, q_hidden.size(-1))
        logits = torch.einsum("bih,bjh->bji", bilinear_q, seq_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # Calculate the class logits
        pooled_output = self.dropout(self.pooler(hidden))
        switch_logits = self.qa_classifier(pooled_output)

        if start_positions is not None and end_positions is not None and switch_list is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduction='none')

            # if no_masking is True, we do not mask the no-answer examples'
            # span losses.
            if self.no_masking is True:
                start_losses = loss_fct(start_logits, start_positions)
                end_losses = loss_fct(end_logits, end_positions)

            else:
                # You care about the span only when switch is 0
                span_mask = (switch_list == 0).type(torch.FloatTensor).cuda()
                start_losses = loss_fct(start_logits, start_positions) * span_mask
                end_losses = loss_fct(end_logits, end_positions) * span_mask

            switch_losses = loss_fct(switch_logits, switch_list)
            assert len(start_losses) == len(end_losses) == len(switch_losses)

            return (self.lambda_scale * (start_losses + end_losses) + switch_losses).mean()

        elif start_positions is None or end_positions is None or switch_list is None:
            return start_logits, end_logits, switch_logits

        else:
            raise NotImplementedError()


class RobertaForQuestionAnsweringConfidence(RobertaPreTrainedModel):

    def __init__(self, config: RobertaConfig, no_masking, lambda_scale=1.0):
        super(RobertaForQuestionAnsweringConfidence, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.num_labels = config.num_labels
        self.no_masking = no_masking
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(
            config.hidden_size, 2)  # [N, L, H] => [N, L, 2]
        self.qa_classifier = nn.Linear(
            config.hidden_size, self.num_labels)  # [N, H] => [N, n_class]
        self.lambda_scale = lambda_scale

        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask,
                start_positions=None, end_positions=None, switch_list=None):
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # Calculate the sequence logits.
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # Calculate the class logits
        pooled_output = self.dropout(pooled_output)
        switch_logits = self.qa_classifier(pooled_output)

        if start_positions is not None and end_positions is not None and switch_list is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(
                ignore_index=ignored_index, reduction='none')

            # if no_masking is True, we do not mask the no-answer examples'
            # span losses.
            if self.no_masking is True:
                start_losses = loss_fct(start_logits, start_positions)
                end_losses = loss_fct(end_logits, end_positions)

            else:
                # You care about the span only when switch is 0
                span_mask = (switch_list == 0).type(torch.FloatTensor).cuda()
                start_losses = loss_fct(
                    start_logits, start_positions) * span_mask
                end_losses = loss_fct(end_logits, end_positions) * span_mask

            switch_losses = loss_fct(switch_logits, switch_list)
            assert len(start_losses) == len(
                end_losses) == len(switch_losses)
            return (self.lambda_scale * (start_losses + end_losses) + switch_losses).mean()

        elif start_positions is None or end_positions is None or switch_list is None:
            return start_logits, end_logits, switch_logits

        else:
            raise NotImplementedError()
