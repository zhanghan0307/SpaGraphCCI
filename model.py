import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import random
import scipy.sparse as sp

class GATLayer(nn.Module):
    """ 图注意力网络（GAT）层"""
    def __init__(self, input_feature, output_feature, dropout, alpha, concat=True):
        """
        初始化 GAT 层的参数。
        Args:
            input_feature: 输入特征的维度（节点特征的维度）。
            output_feature: 输出特征的维度（注意力机制后的特征维度）。
            dropout: 在注意力权重中的 dropout 比例，防止过拟合。
            alpha: LeakyReLU 中的负斜率系数。
            concat: 如果为 True，则在激活函数后连接多头的输出；如果为 False，则取平均。
        """
        super(GATLayer, self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.a = nn.Parameter(torch.empty(size=(2 * output_feature, 1)))
        self.w = nn.Parameter(torch.empty(size=(input_feature, output_feature)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        """
            参数初始化函数，使用 Xavier 均匀分布对参数进行初始化。
        """
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        """
        前向传播函数，计算节点的新的特征表示。
        Args:
            h(torch.Tensor): 输入的节点特征矩阵，尺寸为 (N, input_feature)，其中 N 是节点数量。
            adj:(torch.Tensor)邻接矩阵，尺寸为 (N, N)，表示节点之间的连接关系。

        Returns: h_prime (torch.Tensor) 输出的节点特征矩阵，尺寸为 (N, output_feature)。

        """
        # 线性变换，将输入特征映射到输出特征空间
        # Wh 的尺寸为 (N, output_feature)
        Wh = torch.mm(h, self.w)
        e = self._prepare_attentional_mechanism_input(Wh) # 计算注意力系数矩阵 e，尺寸为 (N, N)
        zero_vec = -9e15 * torch.ones_like(e) # 创建一个值为 -∞ 的张量，用于在 Softmax 前将非邻居的注意力系数置为极小值
        # 使用邻接矩阵掩码，将非邻居的注意力系数设置为极小值，邻居节点保持原值
        # 这样在 Softmax 计算后，非邻居的注意力权重接近于 0
        attention = torch.where(adj > 0, e, zero_vec)  # adj>0的位置使用e对应位置的值替换，其余都为-9e15，这样设定经过Softmax后每个节点对应的行非邻居都会变为0。
        # 对每个节点的邻居节点的注意力系数进行 Softmax 归一化
        # 计算得到的 attention 尺寸为 (N, N)，每一行表示一个节点对其所有邻居的注意力权重
        attention = F.softmax(attention, dim=1)  # 每行做Softmax，相当于每个节点做softmax
        attention = F.dropout(attention, self.dropout, training=self.training) # 在注意力权重上应用 Dropout，防止过拟合
        # 将注意力权重与节点特征进行加权求和，得到新的节点特征表示
        # h_prime 尺寸为 (N, output_feature)
        h_prime = torch.mm(attention, Wh)  # 得到下一层的输入

        if self.concat:
            # 如果 concat 为 True，使用 ELU 激活函数
            return F.elu(h_prime)  # 激活
        else:
            # 如果 concat 为 False，直接返回输出，不使用激活函数（用于最后一层）
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        """
        计算注意力机制中的 e 矩阵，用于表示节点之间的注意力系数（未归一化）。
        Args:
            Wh:(torch.Tensor)输入特征经过线性变换后的矩阵，尺寸为 (N, output_feature)。

        Returns: e (torch.Tensor): 注意力系数矩阵，尺寸为 (N, N)。

        """
        # 计算 Wh1，尺寸为 (N, 1)
        # self.a 的前一半参数（对应于节点 i）
        Wh1 = torch.matmul(Wh, self.a[:self.output_feature, :])  # N*out_size @ out_size*1 = N*1
        # 计算 Wh2，尺寸为 (N, 1)
        # self.a 的后一半参数（对应于节点 j）
        Wh2 = torch.matmul(Wh, self.a[self.output_feature:, :])  # N*1
        # 计算 e 矩阵，尺寸为 (N, N)
        # 将 Wh1 和 Wh2 进行广播相加，得到每对节点之间的注意力系数（未激活）
        e = Wh1 + Wh2.T  # Wh1的每个原始与Wh2的所有元素相加，生成N*N的矩阵
        return self.leakyrelu(e) # 对 e 矩阵应用 LeakyReLU 激活函数

def decode(z):
    """
    使用节点的表示向量来重建邻接矩阵，返回节点之间连接的概率。
    Args:
        z: (torch.Tensor)节点的表示向量矩阵，尺寸为 (N, d)，其中 N 是节点的数量，d 是表示向量的维度。

    Returns:A_pred (torch.Tensor)预测的邻接矩阵，尺寸为 (N, N)，其中每个元素表示节点之间的连接概率。

    """
    # z所有节点的表示向量
    # 使用节点的表示向量 z 计算重建的邻接矩阵。
    # 首先计算节点表示向量的内积：z @ z.T，其中 z.T 是表示向量的转置。
    # 这样得到的矩阵是每个节点对之间的相似性分数。
    # z 的尺寸为 (N, d)，z.t() 的尺寸为 (d, N)，结果为 (N, N)
    # 表示每对节点的关系。
    A_pred = torch.sigmoid(torch.matmul(z, z.t()))
    return A_pred

class GAT(nn.Module):
    """图注意力网络（GAT）模型,该模型结合了节点的注意力机制和特征注意力机制，最终通过多层感知机进行融合。"""
    def __init__(self, input_size, input_feature_size,hidden_size, output_size, dropout, alpha, nheads, concat=True):
        """

        Args:
            input_size: 输入的节点数量。
            input_feature_size: 输入的特征数量。
            hidden_size: 每层 GAT 层的隐藏单元数。
            output_size: GAT 输出的特征维度。
            dropout: Dropout 比例，防止过拟合。
            alpha: LeakyReLU 中的负斜率系数。
            nheads: 多头注意力的数量。
            concat: 是否连接注意力头的输出。
        """
        super(GAT, self).__init__()
        self.dropout = dropout # Dropout 用于防止过拟合
        # 节点注意力层，使用 nheads 个多头注意力机制进行并行计算
        # 对于每个头，GATLayer 负责对节点进行注意力计算
        self.attention = [GATLayer(input_size, hidden_size, dropout=dropout, alpha=alpha, concat=True) for _ in
                          range(nheads)]
        # 将多个头的 GATLayer 加入到模型的模块列表中
        for i, attention in enumerate(self.attention):
            self.add_module('attention_{}'.format(i), attention)

        # 输出的节点注意力层，将多个头的输出整合成最终的输出，输出维度为 output_size
        self.out_att = GATLayer(hidden_size * nheads, output_size, dropout=dropout, alpha=alpha, concat=False)

        # 特征注意力层，类似于节点注意力层，使用 nheads 个多头注意力机制进行并行计算
        self.feature_attention = [GATLayer(input_feature_size, hidden_size, dropout=dropout, alpha=alpha, concat=True) for _ in
                          range(nheads)]
        # 将多个头的特征注意力层加入到模型中
        for i, attention in enumerate(self.feature_attention):
            self.add_module('attention_{}'.format(i), attention)

        # 输出的特征注意力层，输出维度为 output_size
        self.feature_out_att = GATLayer(hidden_size * nheads, output_size, dropout=dropout, alpha=alpha, concat=False)

        # 解码器，用于将节点特征重构为输入的大小
        self.linear_decoder = nn.Linear(output_size, input_size)
        # 特征解码器，用于将特征重构为输入的特征大小
        self.feature_linear_decoder = nn.Linear(output_size, input_feature_size)

        # 多层感知机（MLP），用于将节点特征和特征注意力结果进行融合
        self.MLP = nn.Sequential(
                    nn.Linear(2*output_size, output_size)
                )


    def forward(self, x, feature,adj):
        """
        前向传播函数，计算节点特征、特征的注意力表示，并返回重构结果。
        Args:
            x: 输入的节点特征，尺寸为 (batch_size, input_size)。
            feature: 输入的特征，尺寸为 (batch_size, input_feature_size)。
            adj: 邻接矩阵，尺寸为 (batch_size, input_size, input_size)。

        Returns:
            - reconstruct_adj (torch.Tensor): 重构的邻接矩阵，尺寸为 (batch_size, input_size, input_size)。
            - reconstruct_x (torch.Tensor): 重构的节点特征，尺寸为 (batch_size, input_size)。
            - reconstruct_feature (torch.Tensor): 重构的输入特征，尺寸为 (batch_size, input_feature_size)。

        """
        x = F.dropout(x, self.dropout, training=self.training) # 在输入的节点特征上应用 Dropout
        x = torch.cat([att(x, adj) for att in self.attention], dim=1) # 将每个注意力头的输出拼接在一起
        x = F.dropout(x, self.dropout, training=self.training) # 再次应用 Dropout
        x = F.elu(self.out_att(x, adj)) # 对输出使用 ELU 激活函数

        # 处理输入特征的注意力机制
        feature = F.dropout(feature, self.dropout, training=self.training) # 在输入的特征上应用 Dropout
        feature = torch.cat([att(feature, adj) for att in self.feature_attention], dim=1) # 将每个注意力头的输出拼接在一起
        feature = F.dropout(feature, self.dropout, training=self.training) # 再次应用 Dropout
        feature = F.elu(self.feature_out_att(feature, adj)) # 对输出使用 ELU 激活函数

        # 融合节点和特征注意力结果
        z = torch.cat([x,feature],dim=1) # 拼接节点注意力和特征注意力的结果
        z = self.MLP(z) # 使用 MLP 进行融合，输出维度为 output_size

        # 解码器，重构邻接矩阵、节点特征和输入特征
        reconstruct_adj = decode(z) # 重构邻接矩阵
        reconstruct_x = self.linear_decoder(z) # 重构节点特征
        reconstruct_feature = self.feature_linear_decoder(z) # 重构输入特征

        return reconstruct_adj,reconstruct_x,reconstruct_feature
