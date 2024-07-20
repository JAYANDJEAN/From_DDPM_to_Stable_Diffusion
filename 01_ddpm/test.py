from utils import *
from torch import nn, Tensor
from typing import Optional, List
import math

# n_steps = 3
# dim = 4
# t = torch.randint(0, n_steps, (128,))
# print(t)
# position = torch.arange(0, n_steps).unsqueeze(1)
# print(position)
# div_term = torch.exp(- torch.arange(0, dim, 2) * (math.log(10000.0) / dim))
# pos_embedding = torch.zeros(n_steps, dim)
# pos_embedding[:, 0::2] = torch.sin(position * div_term)
# pos_embedding[:, 1::2] = torch.cos(position * div_term)
#
# print(pos_embedding.shape)
# print(pos_embedding[t].shape)


# 假设你的类别数据是以下形式
labels = torch.tensor([0, 1, 2, 1, 0])  # 这里的batch_size是5

# 设定类别总数num_class
num_class = 3

# 使用torch.nn.functional.one_hot进行one-hot编码
one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_class)

print(one_hot_labels)
