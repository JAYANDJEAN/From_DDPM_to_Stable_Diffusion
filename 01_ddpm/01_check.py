from utils import *
from unet import UNet
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
batch_size = 128
n_step = 1000
n_class = 10


def visual_alpha():
    def compute_ddpm_params(T, beta_start, beta_end):
        betas = np.linspace(beta_start, beta_end, T)
        alphas = 1 - betas
        alphas_bar = np.cumprod(alphas)
        sqrt_alphas_bar = np.sqrt(alphas_bar)
        sqrt_one_minus_alphas_bar = np.sqrt(1 - alphas_bar)
        return sqrt_alphas_bar, sqrt_one_minus_alphas_bar

    T = 1000
    beta_start = 0.0001
    beta_end = 0.02
    alpha, beta = compute_ddpm_params(T, beta_start, beta_end)
    plt.figure(figsize=(12, 8))
    plt.plot(alpha, label='sqrt_alphas_bar')
    plt.plot(beta, label='sqrt_one_minus_alphas_bar', color='orange')
    plt.title('DDPM Parameters')
    plt.legend()
    plt.xlabel('Timesteps')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig('../00_assets/parameters.png')


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
                channel_mults=[1, 2, 2, 2],
                dropout=0.1,
                n_steps=n_step,
                num_class=n_class)
    x_recon = unet(x1, t, y)
    assert x_recon.shape == x1.shape
    print(f"\nnumber of parameters: {sum([p.numel() for p in unet.parameters()])}")


if __name__ == '__main__':
    visual_alpha()
