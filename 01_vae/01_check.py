import torch
from models import VanillaVAE, SDVAE, AttentionBlock, ResidualBlock
import torch.nn as nn


def check_vae():
    x = torch.randn((128, 3, 128, 128))
    vae = VanillaVAE(in_channels=3, image_size=128, latent_dim=256)
    mu, log_var = vae.encode(x)
    print(f"mu shape: {mu.shape}, var shape: {log_var.shape}")
    z = vae.reparameterize(mu, log_var)
    print(f"z shape: {z.shape}")
    recon = vae.decode(z)
    print(f"recons shape: {recon.shape}")

    result = vae(x)
    loss = vae.loss_function(*result, M_N=0.005)
    print(loss)


def check_sdvae():
    x = torch.randn((5, 3, 512, 512))
    vae = SDVAE(in_channels=3, image_size=512, latent_dim=256)
    z = vae.encode(x)
    print(f"z shape: {z.shape}")
    recon = vae.decode(z)
    print(f"recon shape: {recon.shape}")


def check_conv():
    x = torch.randn((4, 3, 512, 512))
    conv = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=0)
    print(conv(x).shape)

    encoder = nn.Sequential(
        nn.Conv2d(3, 128, kernel_size=3, padding=1),
        ResidualBlock(128, 128),
        ResidualBlock(128, 128),
        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
        ResidualBlock(128, 256),
        ResidualBlock(256, 256),
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
        ResidualBlock(256, 512),
        ResidualBlock(512, 512),
        nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
        ResidualBlock(512, 512),
        ResidualBlock(512, 512),
        ResidualBlock(512, 512),
        AttentionBlock(512),
        ResidualBlock(512, 512),
        nn.GroupNorm(32, 512),
        nn.SiLU(),
        nn.Conv2d(512, 8, kernel_size=3, padding=1),
        nn.Conv2d(8, 8, kernel_size=1, padding=0),
    )

    print(encoder(x).shape)


if __name__ == '__main__':
    check_sdvae()
