import torch
from unet import UNet
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda
from utils import *
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets import CIFAR10
from schedulers import *
from utils import *
from unet import UNet
import yaml
from torch import nn
from torch.optim import Adam

torch.manual_seed(0)
batch_size = 128
n_step = 1000
n_class = 10


def check_conv():
    ch_in = 3
    ch_out = 32

    x = torch.rand(batch_size, ch_in, 32, 32)
    conv1 = nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1)
    print(conv1(x).shape)

    conv2 = nn.Conv2d(ch_in, ch_out, 3, stride=1, padding=1)
    trans = nn.ConvTranspose2d(ch_out, ch_out, 5, stride=2, padding=2, output_padding=1)
    print(trans(conv2(x)).shape)


def check_unet_output():
    t = torch.randint(0, n_step, (batch_size,))
    y = torch.randint(0, n_class, (batch_size,))
    x1 = torch.rand(batch_size, 3, 32, 32)
    unet = UNet(channel_img=3,
                channel_base=128,
                channel_mults=[1, 1, 2, 4],
                dropout=0.1,
                n_steps=n_step,
                num_class=n_class)
    x_recon = unet(x1, t, y)
    assert x_recon.shape == x1.shape
    print(f"\nnumber of parameters: {sum([p.numel() for p in unet.parameters()])}")


if __name__ == '__main__':
    check_unet_output()
