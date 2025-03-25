import torch
import torch.nn.functional as F
from torch import nn
from torch import nn
from transformers import RobertaConfig, RobertaModel







class CNN(nn.Module):
    """Network for fine-tuning on IL properties datasets"""
    def __init__(self,
                 dropout,
                 embed_size,
                 output_size=1,
                 num_filters=(100, 200, 200, 200, 200, 100, 100),
                 ngram_filter_sizes=(1, 2, 3, 4, 5, 6, 7),
                 IL_num_filters=(100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160),
                 IL_ngram_filter_sizes=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15)):
        super(CNN, self).__init__()
        
        self.num_filters = num_filters
        self.IL_num_filters = IL_num_filters
        
        self.IL_textcnn = nn.ModuleList([nn.Conv1d(in_channels=embed_size, out_channels=nf, kernel_size=ks)
                                        for nf, ks in zip(IL_num_filters, IL_ngram_filter_sizes)])
        self.output = nn.Linear(sum(IL_num_filters), embed_size)


    def forward(self, IL_src_nd):
        IL_encoded = IL_src_nd.permute(1,2,0)

        IL_textcnn_out = [F.relu(conv(IL_encoded)) for conv in self.IL_textcnn]
        IL_textcnn_out = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in IL_textcnn_out]  # Max pooling
        IL_textcnn_out = torch.cat(IL_textcnn_out, 1)  # Concatenate all the pooled features
        # input_vecs = torch.cat((IL_textcnn_out, T.view(-1, 1), P.view(-1, 1)), dim=1)
        input_vecs = IL_textcnn_out
        # input_vecs = torch.cat((self.lin(IL_textcnn_out), T.view(-1, 1)), dim=1)

        out = self.output(input_vecs.float())

        # out = self.output(IL_encoded[:,0,:].float())

        return out



import numpy as np
import math

class ILBERT_T(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float):
        super().__init__()
        self.model_type = 'RoBERTa'

        config = RobertaConfig(
            vocab_size=ntoken,  # 词汇表大小
            hidden_size=d_model,  # 隐藏层维度
            num_hidden_layers=nlayers,  # Transformer层数
            num_attention_heads=nhead,  # 注意力头数
            intermediate_size=d_hid,  # 中间层维度
            hidden_dropout_prob=dropout,  # 隐藏层dropout
            attention_probs_dropout_prob=dropout,
            output_attentions=True,  # 确保返回注意力权重
        )
        self.roberta = RobertaModel(config)
        self.CNN = CNN(embed_size=d_model, dropout=dropout)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model+1, d_model//2),
            nn.Softplus(),
            # nn.Dropout(dropout),
            nn.Linear(d_model//2, 1)
        )


    def forward(self, input):
        """
        Args:
            input: 输入数据，格式为(src, labels),其中src的形状应为[batch_size, seq_len]
        Returns:
            output: 模型输出
            attentions: 注意力权重
        """
        x, T = input

        # 生成attention_mask，对于src中不为0的位置，mask值为1；否则为0
        attention_mask = (x != 0).long()

        # 将输入和attention_mask传递给RoBERTa模型
        outputs = self.roberta(input_ids=x.long(), attention_mask=attention_mask)

        last_hidden_states = outputs.last_hidden_state
        last_hidden_states = last_hidden_states.permute(1, 0, 2)  # 调整维度以符合CNN的输入要求

        output = self.CNN(last_hidden_states)
        T=T.view(-1, 1)


        output = self.pred_head(torch.cat((output, T.float()), dim=1))
        return output.float()



