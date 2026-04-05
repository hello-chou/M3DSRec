from collections import OrderedDict
import torch
import torch.nn.functional as F
from teacher import TEACHER
from student import  STUDENT
from recbole.model.abstract_recommender import SequentialRecommender


class alltoone(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)


        self.stage = config['stage']
        self.temperature = config['temperature']
        self.loss_method= config['loss_method']

        self.klweight = config['klweight']
        if self.stage == "pretrain":
            self.teacher_network = TEACHER(config, dataset)
        elif self.stage == "finetune":
            self.student_network = STUDENT(config, dataset)
            self.teacher_network = TEACHER(config, dataset)
        if self.stage != "pretrain":
            save_info = torch.load(config["warm_up"], weights_only=False)
            new_state_dict = OrderedDict()
            for name, val in save_info['state_dict'].items():
                print(name)
            self.teacher_network.load_state_dict(new_state_dict, strict=False)
            for _ in self.teacher_network.parameters():
                _.requires_grad = False

    def _compute_seq_embeddings(self, item_seq, item_seq_len):

        if self.stage == "pretrain":
            output_teacher, fuzhu_teacher = self.teacher_network._compute_seq_embeddings(item_seq, item_seq_len)
            return output_teacher, fuzhu_teacher
        elif self.stage == "finetune":
            output_student, fuzhu_student = self.student_network._compute_seq_embeddings(item_seq, item_seq_len)
            return output_student, fuzhu_student





    def ke_loss(self, teacher, student, temperature):
        beta_logits_teacher = teacher / temperature
        beta_logits_student = student / temperature
        p_teacher = F.softmax(beta_logits_teacher, dim=-1)
        loss = -(p_teacher * F.log_softmax(beta_logits_student, dim=-1)).sum(dim=-1).mean()
        return loss * (temperature * temperature)

    def calculate_loss(self, interaction):

        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        if self.stage == "pretrain":
            loss, _ = self.teacher_network.calculate_loss(interaction)
        elif self.stage == "finetune":
            loss, _ = self.student_network.calculate_loss(interaction)
            self.teacher_network.eval()
            kl_loss_output = self.ke_loss(self.teacher_network._compute_seq_embeddings(item_seq, item_seq_len)[0],
                                          self.student_network._compute_seq_embeddings(item_seq, item_seq_len)[0],
                                          temperature=self.temperature)
            kl_loss_input = self.ke_loss((self.teacher_network._compute_seq_embeddings(item_seq, item_seq_len)[1][
                                              0] +
                                          self.teacher_network._compute_seq_embeddings(item_seq, item_seq_len)[1][
                                              1] +
                                          self.teacher_network._compute_seq_embeddings(item_seq, item_seq_len)[1][
                                              -1]),
                                         self.student_network._compute_seq_embeddings(item_seq, item_seq_len)[1],
                                         temperature=self.temperature)
            kl_loss_logits = self.ke_loss(self.teacher_network.calculate_loss(interaction)[1],
                                          self.student_network.calculate_loss(interaction)[1],
                                          temperature=self.temperature)

            loss = loss + self.klweight * (kl_loss_output + kl_loss_input + kl_loss_logits)


        return loss

    def full_sort_predict(self, interaction):
        if self.stage == "pretrain":
            scores = self.teacher_network.full_sort_predict(interaction)
        elif self.stage == "finetune":
            scores = self.student_network.full_sort_predict(interaction)
        return scores

    def predict(self, interaction):
        if self.stage == "pretrain":
            scores = self.teacher_network.predict(interaction)
        elif self.stage == "finetune":
            scores = self.student_network.predict(interaction)
        return scores

