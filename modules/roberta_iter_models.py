import torch
from torch import nn
import logging
from transformers.modeling_roberta import RobertaConfig, RobertaPreTrainedModel, RobertaModel

from modules import layers

logger = logging.getLogger(__name__)


class IterRobertaPreTrainedConfig(RobertaConfig):
    added_configs = [
        'query_dropout', 'cls_type', 'sr_query_dropout', 'lm_query_dropout',
        'z_step', 'pos_emb_size', 'weight_typing'
    ]

    def __init__(self, query_dropout=0.1, cls_type=0,
                 sr_query_dropout=0.1, lm_query_dropout=0.1,
                 pos_emb_size=200, z_step=0, weight_typing=True, **kwargs):
        super().__init__(**kwargs)

        self.query_dropout = query_dropout
        self.cls_type = cls_type
        self.sr_query_dropout = sr_query_dropout
        self.lm_query_dropout = lm_query_dropout
        self.pos_emb_size = pos_emb_size
        self.z_step = z_step
        self.weight_typing = weight_typing

    def expand_configs(self, *args):
        self.added_configs.extend(list(args))


class IterRobertaModel(RobertaPreTrainedModel):
    config_class = IterRobertaPreTrainedConfig
    model_prefix = 'iter_roberta'

    def __init__(self, config: IterRobertaPreTrainedConfig):
        super().__init__(config)

        self.config = config
        self.roberta = RobertaModel(config)

        self.query = layers.MultiHeadAlignedTokenAttention(
            config,
            attn_dropout_p=config.query_dropout,
            dropout_p=config.query_dropout
        )

        self.z_step = config.z_step

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                **kwargs):
        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        batch, sent_num, seq_len = sent_word_mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        sent_word_hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)

        q_vec = seq_output[:, :1]  # <s>
        for _step in range(self.z_step):
            if _step == 0:
                _aligned = False
            else:
                _aligned = True
            q_vec = self.query(q_vec, sent_word_hidden, sent_word_mask, aligned=_aligned, residual=False)
            if _step == 0:
                q_vec = q_vec.squeeze(1)

        hidden_sent = q_vec
        assert hidden_sent.size() == (batch, sent_num, seq_output.size(-1))

        return hidden_sent, seq_output, sent_word_hidden


class IterRobertaModelForRetrievalV3(IterRobertaModel):
    model_prefix = 'iter_roberta_retrieval_v3'

    def __init__(self, config: IterRobertaPreTrainedConfig):
        super().__init__(config)

        self.retrieval_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.retrieval_k = nn.Linear(config.hidden_size, config.hidden_size)

        self.retrieval_o = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.iter_roberta_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        batch, seq_len = input_ids.size()

        # `token_type_ids`: [0,0,0,1,1,1,1,0,0,0]
        # `attention_mask`: [1,1,1,1,1,1,1,0,0,0]
        # `1` for true token and `0` for mask
        question_mask = (1 - token_type_ids) * attention_mask
        passage_mask = token_type_ids * attention_mask

        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask)[0]
                                  # token_type_ids=token_type_ids)[0]

        cls_h = seq_output[:, :1]
        question_mask = question_mask.to(seq_output.dtype)
        passage_mask = passage_mask.to(seq_output.dtype)

        q_hidden = self.query(cls_h, seq_output.unsqueeze(1), 1 - question_mask.unsqueeze(1),
                              aligned=True, residual=False).view(batch, seq_output.size(-1))

        retrieve_q = self.retrieval_q(q_hidden)
        retrieve_k = self.retrieval_k(seq_output)
        retrieve_s = torch.einsum("bh,bsh->bs", retrieve_q, retrieve_k)
        retrieve_a = torch.softmax(retrieve_s + (1 - passage_mask) * -10000.0, dim=1)
        retrieve_o = self.layer_norm(self.iter_roberta_dropout(self.retrieval_o(torch.cat([
            q_hidden, torch.einsum("bs,bsh->bh", retrieve_a, seq_output)
        ], dim=-1))))

        return retrieve_o
