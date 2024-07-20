import torch
from unet import UNet
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda
from utils import *

batch_size = 128
n_step = 1000
n_class = 10
n_time_embed = 100


def check_data():
    transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
    dataset = CIFAR10(root="../00_assets/datasets", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    show_first_batch(loader)


def check_unet_output():
    t = torch.randint(0, n_step, (batch_size,))
    y = torch.randint(0, n_class, (batch_size,))

    x0 = torch.rand(batch_size, 1, 28, 28)
    unet = UNet(channels=[1, 10, 20, 40, 64],
                dropout=0.1,
                time_emb_dim=n_time_embed,
                n_steps=n_step,
                num_class=n_class)
    x_recon = unet(x0, t, None)
    assert x_recon.shape == x0.shape

    x1 = torch.rand(batch_size, 3, 32, 35)
    unet = UNet(channels=[3, 32, 64, 128],
                dropout=0.1,
                time_emb_dim=100,
                n_steps=n_step,
                num_class=n_class)
    x_recon = unet(x1, t, y)
    assert x_recon.shape == x1.shape


if __name__ == '__main__':
    check_data()
