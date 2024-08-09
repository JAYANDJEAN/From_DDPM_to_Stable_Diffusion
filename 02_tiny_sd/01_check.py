from utils import *
from diffusion import Diffusion
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

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
    diffusion = Diffusion(channel_img=3, channel_multy=[1, 2, 4, 8], num_class=n_class)
    x_recon = diffusion(x1, t, y)
    assert x_recon.shape == x1.shape
    print(f"\nnumber of parameters: {sum([p.numel() for p in diffusion.parameters()])}")


def check_animal_faces():
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])
    dataset = datasets.ImageFolder(root='../00_assets/datasets/afhq/train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=36, shuffle=True, num_workers=4)

    for images, labels in dataloader:
        print(images.shape)
        save_image(tensor=denormalize(images.clone(), means, stds),
                   fp=f"../00_assets/image/animal_faces.png",
                   nrow=6)
        break


def check_warmup():
    diffusion = Diffusion(channel_img=3, channel_multy=[1, 2, 4, 8], num_class=n_class)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = CosineWarmupScheduler(optimizer=optimizer,
                                      warmup_epochs=7,
                                      max_lr=1e-4,
                                      total_epochs=70)

    for epoch in range(70):
        optimizer.step()
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"epoch: {epoch}, current_lr: {current_lr:.6f}")
        scheduler.step()


if __name__ == '__main__':
    check_warmup()
