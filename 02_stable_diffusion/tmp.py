import torch
import torch.nn as nn

y = torch.as_tensor([5] * 10)
print(y)
y_emb = nn.functional.one_hot(y, num_classes=10).float()
print(y_emb)
