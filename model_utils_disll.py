import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.layers import TransformerEncoder
class MemoryBank(nn.Module):
    def __init__(self, K=1024, D=256):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(K, D))
        self.ptr = 0

    def update(self, features):
        batch = features.detach()
        ptr = self.ptr
        self.memory[ptr:ptr + len(batch)] = 0.9 * self.memory[ptr:ptr + len(batch)] + 0.1 * batch
        self.ptr = (ptr + len(batch)) % self.memory.size(0)

    def query(self, query_vec, topk=5):
        # 基于相似度的记忆检索
        sim = torch.matmul(query_vec, self.memory.T)  # (B,S,K)
        topk_idx = torch.topk(sim, k=topk, dim=-1)[1]  # (B,S,topk)
        return self.memory[topk_idx]  # (B,S,topk,D)


class MemoryEnhancedKMeans(nn.Module):
        def __init__(self, K=1024, D=256, k=10, max_iters=10):
            super().__init__()
            self.memory_bank = MemoryBank(K, D)
            self.k = k
            self.max_iters = max_iters
            self.D = D


            self.register_buffer('ones', torch.ones(1, 1, 1, dtype=torch.float32))

        def forward(self, x):
            memory_ctx = self.memory_bank.query(x)

            x_expanded = x.unsqueeze(2)
            attn = torch.matmul(x_expanded, memory_ctx.transpose(-1, -2))
            attn = F.softmax(attn.squeeze(2), dim=-1)
            x_enhanced = x + torch.einsum('bsk,bskd->bsd', attn, memory_ctx)

            return self.inplace_cluster(x_enhanced, self.k, self.max_iters)

        def inplace_cluster(self, x, k, max_iters):
            B, S, D = x.shape
            device = x.device

            buffer_centroids = torch.empty((B, k, D), device=x.device, dtype=x.dtype)
            buffer_sums = torch.empty((B, k, D), device=x.device, dtype=x.dtype)
            buffer_counts = torch.empty((B, k, 1), device=x.device, dtype=x.dtype)

            indices = torch.randint(0, S, (B, k), device=device)
            batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
            buffer_centroids.copy_(x[batch_indices, indices])

            x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)


            current_ones = self.ones.expand(B, S, 1).to(x.device)

            for _ in range(max_iters):
                dots = torch.matmul(x, buffer_centroids.transpose(1, 2))
                c_norm_sq = (buffer_centroids ** 2).sum(dim=-1)
                distances = x_norm_sq - 2 * dots + c_norm_sq.unsqueeze(1)

                assignments = torch.argmin(distances, dim=-1)

                buffer_sums.zero_()
                buffer_counts.zero_()
                buffer_sums.scatter_add_(1, assignments.unsqueeze(-1).expand(-1, -1, D), x)

                buffer_counts.scatter_add_(1, assignments.unsqueeze(-1), current_ones)

                buffer_centroids.copy_(buffer_sums)
                buffer_centroids.div_(buffer_counts + 1e-8)

            expand_idx = assignments.unsqueeze(-1).expand(-1, -1, D)
            quantized = torch.gather(buffer_centroids, dim=1, index=expand_idx)

            return quantized


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

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.adjust = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.LayerNorm_t = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout_t = nn.Dropout(self.hidden_dropout_prob)
        self.LayerNorm_v = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout_v = nn.Dropout(self.hidden_dropout_prob)

        self.memory_enhanced_kmeans_t = MemoryEnhancedKMeans(256, config['hidden_size'], 10, 10)
        self.memory_enhanced_kmeans_v = MemoryEnhancedKMeans(256, config['hidden_size'], 10, 10)
        if self.modelmethod=='sasrec':
            self.trm_model = TransformerEncoder(
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                hidden_size=self.hidden_size,
                inner_size=self.inner_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                hidden_act=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps,
            )
            self.trm_model_text = TransformerEncoder(
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                hidden_size=self.hidden_size,
                inner_size=self.inner_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                hidden_act=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps,
            )
            self.trm_model_image = TransformerEncoder(
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
            self.mamba_layers = nn.ModuleList([
                MambaLayer(
                    d_model=self.hidden_size,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand,
                    dropout=self.dropout_prob,
                    num_layers=self.mamba_layer,
                ) for _ in range(self.mamba_layer)
            ])
            self.mamba_layers_t = nn.ModuleList([
                MambaLayer(
                    d_model=self.hidden_size,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand,
                    dropout=self.dropout_prob,
                    num_layers=self.mamba_layer,
                ) for _ in range(self.mamba_layer)
            ])
            self.mamba_layers_v = nn.ModuleList([
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
            self.tokenMixer_id = PreNormResidual(self.hidden_size,
                                                 FeedForwardmixer(self.max_seq_length, expansion_factor,
                                                             self.hidden_dropout_prob,
                                                             chan_first))
            self.channelMixer_id = PreNormResidual(self.hidden_size,
                                                   FeedForwardmixer(self.hidden_size, expansion_factor,
                                                               self.hidden_dropout_prob))

            self.tokenMixer_t = PreNormResidual(self.hidden_size,
                                                FeedForwardmixer(self.max_seq_length, expansion_factor,
                                                            self.hidden_dropout_prob,
                                                            chan_first))
            self.channelMixer_t = PreNormResidual(self.hidden_size,
                                                  FeedForwardmixer(self.hidden_size, expansion_factor,
                                                              self.hidden_dropout_prob))

            self.tokenMixer_v = PreNormResidual(self.hidden_size,
                                                FeedForwardmixer(self.max_seq_length, expansion_factor,
                                                            self.hidden_dropout_prob,
                                                            chan_first))
            self.channelMixer_v = PreNormResidual(self.hidden_size,
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
