import copy
from model_utils_disll import Transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """

    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """

    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training)  # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)]  # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)



class TEACHER(Transformer):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.clxishu=config['clxishu']
        self.mamba_layer=config['mamba_layer']
        self.mixer_layers=config['mixer_layers']


        self.plm_embedding = copy.deepcopy(dataset.plm_embedding)
        self.img_embedding = copy.deepcopy(dataset.img_embedding)

        self.text_adaptor = MoEAdaptorLayer(
            config['n_exps'],
            [config['plm_size'], config['hidden_size']],
            config['adaptor_dropout_prob'],
            self.max_seq_length
        )
        self.img_adaptor = MoEAdaptorLayer(
            config['n_exps'],
            [config['img_size'], config['hidden_size']],
            config['adaptor_dropout_prob'],
            self.max_seq_length
        )

        self.img_alpha = torch.nn.Parameter(torch.tensor([0.]))
        self.text_beta = torch.nn.Parameter(torch.tensor([0.]))

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    def forward(self, item_seq, item_emb, item_seq_len):

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        if self.modelmethod == 'sasrec':
            position_embedding = self.position_embedding(position_ids)
        dec_input_emb = item_emb
        item_id_embeddings = self.item_embedding(item_seq)
        item_id_embeddings_ks=item_id_embeddings
        t = dec_input_emb[:, 0, :, :]
        v = dec_input_emb[:, 1, :, :]


        sum_id = item_id_embeddings.sum(dim=1, keepdim=True) #MCB

        t_expanded = t * sum_id #MCB
        v_expanded = v * sum_id #MCB

        t_center = self.memory_enhanced_kmeans_t(t) #MEDC
        t = t_center + t_expanded
        v_center = self.memory_enhanced_kmeans_v(v) #MEDC
        v = v_center + v_expanded



        if self.modelmethod == 'sasrec':
            t = self.LayerNorm_t(t + position_embedding)
            t = self.dropout_t(t)
            v = self.LayerNorm_v(v + position_embedding)
            v = self.dropout_v(v)
            item_id_embeddings = self.LayerNorm(item_id_embeddings + position_embedding)
            item_id_embeddings = self.dropout(item_id_embeddings)
            tgt_attn_mask = self.get_attention_mask(item_seq)


            id = self.trm_model(
                item_id_embeddings, tgt_attn_mask, output_all_encoded_layers=True
            )
            t = self.trm_model_text(
                t, tgt_attn_mask, output_all_encoded_layers=True
            )
            v = self.trm_model_image(
                v, tgt_attn_mask, output_all_encoded_layers=True
            )
            al = self.adjust(torch.cat([id[-1], t[-1], v[-1]], dim=-1))
        elif self.modelmethod == 'mamba':
            t = self.LayerNorm_t(t )
            t = self.dropout_t(t)
            v = self.LayerNorm_v(v )
            v = self.dropout_v(v)
            item_id_embeddings = self.LayerNorm(item_id_embeddings )
            item_id_embeddings = self.dropout(item_id_embeddings)
            for i in range(self.mamba_layer):
                item_id_embeddings = self.mamba_layers[i](item_id_embeddings)
                t = self.mamba_layers_t[i](t)
                v = self.mamba_layers_v[i](v)
            al = self.adjust(torch.cat([item_id_embeddings, t, v], dim=-1))
        elif self.modelmethod == 'mixer':
            t = self.LayerNorm_t(t)
            t = self.dropout_t(t)
            v = self.LayerNorm_v(v)
            v = self.dropout_v(v)
            item_id_embeddings = self.LayerNorm(item_id_embeddings)
            item_id_embeddings = self.dropout(item_id_embeddings)
            for i in range(self.mixer_layers):
                t = self.tokenMixer_t(t)
                t = self.channelMixer_t(t)
                v = self.tokenMixer_v(v)
                v = self.channelMixer_v(v)
                item_id_embeddings = self.tokenMixer_id(item_id_embeddings)
                item_id_embeddings = self.channelMixer_id(item_id_embeddings)
            al = self.adjust(torch.cat([item_id_embeddings, t, v], dim=-1))
        al = self.gather_indexes(al, item_seq_len - 1)
        al = self.LayerNorm(al)
        al = self.dropout(al)
        if self.modelmethod == 'sasrec':
            return al, [dec_input_emb[:, 0, :, :], dec_input_emb[:, 1, :, :], t[-1], v[-1], t_center, v_center,item_id_embeddings_ks]
        elif self.modelmethod == 'mamba':
            return al, [dec_input_emb[:, 0, :, :], dec_input_emb[:, 1, :, :], t, v, t_center, v_center,item_id_embeddings_ks]
        elif self.modelmethod == 'mixer':
            return al, [dec_input_emb[:, 0, :, :], dec_input_emb[:, 1, :, :], t, v, t_center, v_center,item_id_embeddings_ks]

    def _compute_seq_embeddings(self, item_seq, item_seq_len):
        text_emb = self.text_adaptor(self.plm_embedding(item_seq))
        img_emb = self.img_adaptor(self.img_embedding(item_seq))


        item_emb_list =  []
        item_emb_list.append(text_emb)  # append [BxLxD]
        item_emb_list.append(img_emb)  # append [BxLxD]
        item_emb_list = torch.stack(item_emb_list, dim=1)  # [BxMxLxD]

        all, fuzhu = self.forward(
            item_seq=item_seq,
            item_emb=item_emb_list,
            item_seq_len=item_seq_len,
        )

        return all, fuzhu


    def _compute_dynamic_fused_logits(self, seq_output, text_emb, img_emb):
        text_logits = torch.matmul(seq_output, text_emb.transpose(0, 1))  # [BxB]
        img_logits = torch.matmul(seq_output, img_emb.transpose(0, 1))  # [BxB]
        id_logits = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
        agg_logits = id_logits + self.text_beta * text_logits + self.img_alpha * img_logits
        agg_logits = agg_logits
        return agg_logits

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, fuzhu = self._compute_seq_embeddings(item_seq, item_seq_len)

        test_text_emb = self.text_adaptor(self.plm_embedding.weight)
        test_img_emb = self.img_adaptor(self.img_embedding.weight)
        logits = self._compute_dynamic_fused_logits(seq_output, test_text_emb, test_img_emb)
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items)


        loss_tv = 1 - cosine_similarity(fuzhu[2], fuzhu[3], dim=1).mean()
        loss_tv2 = 1 - cosine_similarity(fuzhu[0], fuzhu[1], dim=1).mean()
        loss_tv3 = 1 - cosine_similarity(fuzhu[4], fuzhu[5], dim=1).mean()
        loss = loss + self.clxishu * (loss_tv + loss_tv2 + loss_tv3)
        return loss, logits

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output, fuzhu = self._compute_seq_embeddings(item_seq, item_seq_len)
        test_text_emb = self.text_adaptor(self.plm_embedding.weight)
        test_img_emb = self.img_adaptor(self.img_embedding.weight)
        scores = self._compute_dynamic_fused_logits(seq_output, test_text_emb, test_img_emb)

        return scores
