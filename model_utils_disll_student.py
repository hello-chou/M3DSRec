import torch.nn as nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.layers import TransformerEncoder
from mamba_ssm import Mamba


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(

            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    def forward(self, input_tensor):
        hidden_states = self.mamba(input_tensor)
        if self.num_layers == 1:
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        hidden_states = self.ffn(hidden_states)
        return hidden_states

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.mamba(input_tensor)
        if self.num_layers == 1:
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        return hidden_states

class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

from functools import partial

pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForwardmixer(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )





class Transformer(SequentialRecommender):

    def __init__(self, config, dataset):
        super(Transformer, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.stage = config['stage']
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        self.modelmethod = config['modelmethod']

        self.mamba_layer = config['mamba_layer']
        self.mix_layers = config['mix_layers']
        self.dropout_prob = config["dropout_prob"]
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]


        if self.stage == "finetune":
            self.item_embedding_student = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
            self.LayerNorm_student = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
            self.dropout_student = nn.Dropout(self.hidden_dropout_prob)
            
            if self.modelmethod=='sasrec':
                self.position_embedding_student = nn.Embedding(self.max_seq_length, self.hidden_size)
                self.trm_model_student = TransformerEncoder(
                    n_layers=self.n_layers,
                    n_heads=self.n_heads,
                    hidden_size=self.hidden_size,
                    inner_size=self.inner_size,
                    hidden_dropout_prob=self.hidden_dropout_prob,
                    attn_dropout_prob=self.attn_dropout_prob,
                    hidden_act=self.hidden_act,
                    layer_norm_eps=self.layer_norm_eps,
                )
            elif self.modelmethod=='mamba':
                self.mamba_layers_student = nn.ModuleList([
                MambaLayer(
                    d_model=self.hidden_size,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand,
                    dropout=self.dropout_prob,
                    num_layers=self.mamba_layer,
                ) for _ in range(self.mamba_layer)
                    ])

            elif self.modelmethod=='mixer':
                expansion_factor = 4
                chan_first = partial(nn.Conv1d, kernel_size=1)
                chan_last = nn.Linear
                self.tokenMixer_student = PreNormResidual(self.hidden_size,
                                                    FeedForwardmixer(self.max_seq_length, expansion_factor,
                                                                self.hidden_dropout_prob,
                                                                chan_first))
                self.channelMixer_student = PreNormResidual(self.hidden_size,
                                                    FeedForwardmixer(self.hidden_size, expansion_factor,
                                                                self.hidden_dropout_prob))

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
