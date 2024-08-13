import torch
from vit_pytorch import ViT

v = ViT(
    image_size=512,
    patch_size=32,
    num_classes=3,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=1024,
    dropout=0.1,
    emb_dropout=0.1
)

img = torch.randn(12, 3, 512, 512)

preds = v(img)  # (1, 1000)
print(preds.shape)
