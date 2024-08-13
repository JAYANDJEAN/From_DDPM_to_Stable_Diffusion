from utils import *
from diffusion import Diffusion
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import yaml

torch.manual_seed(0)
batch_size = 128
n_step = 1000
n_class = 10

with open('../00_assets/yml/tiny_sd_direct.yml', 'r') as file:
    config = yaml.safe_load(file)


def check_animal_faces():
    dataloader = animal_faces_loader(config['batch_size'], config['img_size'])

    for images, labels in dataloader:
        print(images.shape)
        save_image(tensor=denormalize(images.clone()),
                   fp=f"../00_assets/image/animal_faces.png",
                   nrow=6,
                   padding=0)
        break


def visual_alpha():
    betas = np.linspace(config['beta_1'], config['beta_T'], config['T'])
    alphas = 1 - betas
    alphas_bar = np.cumprod(alphas)
    sqrt_alphas_bar = np.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = np.sqrt(1 - alphas_bar)
    plt.figure(figsize=(12, 8))
    plt.plot(sqrt_alphas_bar, label='sqrt_alphas_bar')
    plt.plot(sqrt_one_minus_alphas_bar, label='sqrt_one_minus_alphas_bar', color='orange')
    plt.title('DDPM Parameters')
    plt.legend()
    plt.xlabel('Timesteps')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig('../00_assets/image/parameters.png')


def check_conv():
    ch_in = 3
    ch_out = 32

    x = torch.rand(batch_size, ch_in, 32, 32)
    conv1 = nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1)
    print(conv1(x).shape)

    conv2 = nn.Conv2d(ch_in, ch_out, 3, stride=1, padding=1)
    trans = nn.ConvTranspose2d(ch_out, ch_out, 5, stride=2, padding=2, output_padding=1)
    print(trans(conv2(x)).shape)


def check_diffusion_output():
    t = torch.randint(0, n_step, (batch_size,))
    y = torch.randint(0, n_class, (batch_size,))
    x1 = torch.rand(batch_size, 3, 64, 64)
    diffusion = Diffusion(channel_img=config['img_channel'], channel_base=config['channel'],
                          num_class=config['num_class'], channel_multy=config['channel_multy'],
                          dropout=config['dropout'])
    x_recon = diffusion(x1, t, y)
    assert x_recon.shape == x1.shape
    print(f"\nnumber of parameters: {sum([p.numel() for p in diffusion.parameters()])}")


def check_warmup():
    diffusion = Diffusion(channel_img=config['img_channel'], channel_base=config['channel'],
                          num_class=config['num_class'], channel_multy=config['channel_multy'],
                          dropout=config['dropout'])
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = CosineWarmupScheduler(optimizer=optimizer,
                                      warmup_epochs=config['epoch'] // 7,
                                      max_lr=config['max_lr'],
                                      total_epochs=config['epoch'])

    for epoch in range(config['epoch']):
        optimizer.step()
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"epoch: {epoch:03}, current_lr: {current_lr:.7f}")
        scheduler.step()


if __name__ == '__main__':
    check_warmup()
