from model_utils_disll_student import Transformer
import torch

class STUDENT(Transformer):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.modelmethod=  config['modelmethod']
        self.clxishu=config['clxishu']
        self.mamba_layer=config['mamba_layer']
        self.mixer_layers=config['mixer_layers']


    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask


    def forward(self, item_seq, item_seq_len):
        
        if self.modelmethod=='sasrec':
            position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
            position_embedding = self.position_embedding_student(position_ids)  # [LxD]
            item_id_embeddings = self.item_embedding_student(item_seq)
            item_id_embeddings_kl = item_id_embeddings
            item_id_embeddings = item_id_embeddings + position_embedding
            item_id_embeddings = self.LayerNorm_student(item_id_embeddings)
            item_id_embeddings = self.dropout_student(item_id_embeddings)
            tgt_attn_mask = self.get_attention_mask(item_seq)
            item_id_embeddings = self.trm_model_student(
                item_id_embeddings, tgt_attn_mask, output_all_encoded_layers=True
            )
            al = self.gather_indexes(item_id_embeddings[-1], item_seq_len - 1)
        elif self.modelmethod == 'mamba':
            item_id_embeddings = self.item_embedding_student(item_seq)
            item_id_embeddings_kl = item_id_embeddings
            item_id_embeddings = self.LayerNorm_student(item_id_embeddings)
            item_id_embeddings = self.dropout_student(item_id_embeddings)
            for i in range(self.mamba_layer):
                item_id_embeddings=self.mamba_layers_student[i](item_id_embeddings)
            al = self.gather_indexes(item_id_embeddings, item_seq_len - 1)
        elif self.modelmethod == 'mixer':
            item_id_embeddings = self.item_embedding_student(item_seq)
            item_id_embeddings_kl = item_id_embeddings
            item_id_embeddings = self.LayerNorm_student(item_id_embeddings)
            item_id_embeddings = self.dropout_student(item_id_embeddings)
            for i in range(self.mixer_layers):
                item_id_embeddings = self.tokenMixer_student(item_id_embeddings)
                item_id_embeddings = self.channelMixer_student(item_id_embeddings)
            al = self.gather_indexes(item_id_embeddings, item_seq_len - 1)
        al = self.LayerNorm_student(al)

        return al, item_id_embeddings_kl  # [BxD], []

    def _compute_seq_embeddings(self, item_seq, item_seq_len):

        all = self.forward(
            item_seq=item_seq,
            item_seq_len=item_seq_len,
        )
        return all


    def calculate_loss(self, interaction):

        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output, fuzhu = self._compute_seq_embeddings(item_seq, item_seq_len)
        logits = torch.matmul(seq_output, self.item_embedding_student.weight.transpose(0, 1))
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items)

        return loss, logits

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output, _ = self._compute_seq_embeddings(item_seq, item_seq_len)
        scores = torch.matmul(seq_output, self.item_embedding_student.weight.transpose(0, 1))
        return scores


