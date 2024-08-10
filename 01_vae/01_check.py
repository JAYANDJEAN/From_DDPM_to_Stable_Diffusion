import torch
from models import VanillaVAE, SDVAE, AttentionBlock, ResidualBlock

from diffusers.models import AutoencoderKL

from torch import nn

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


def denormalize(tensor, mean, std):
    device = tensor.device
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(device)
    tensor = tensor
    return tensor * std + mean


def check_hf_vae():
    """
    对比了一下，stabilityai/sdxl-vae貌似更好
    """
    sdxl_vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    sdxl_vae_fix = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])
    dataset = datasets.ImageFolder(root='../00_assets/datasets/afhq/train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    _, batch = next(enumerate(dataloader))
    save_image(tensor=denormalize(batch[0].clone(), means, stds),
               fp=f"../00_assets/image/animal_faces_raw.png",
               nrow=4)
    with torch.no_grad():
        latents = sdxl_vae.encode(batch[0]).latent_dist.sample()
        print(latents.shape)
        decoded_image = sdxl_vae.decode(latents).sample
        print(decoded_image.shape)
        save_image(tensor=denormalize(decoded_image.clone(), means, stds),
                   fp=f"../00_assets/image/animal_faces_sdxl_vae_latent.png",
                   nrow=4)

        latents = sdxl_vae_fix.encode(batch[0]).latent_dist.sample()
        print(latents.shape)
        decoded_image = sdxl_vae_fix.decode(latents).sample
        print(decoded_image.shape)
        save_image(tensor=denormalize(decoded_image.clone(), means, stds),
                   fp=f"../00_assets/image/animal_faces_sdxl_vae_fix_latent.png",
                   nrow=4)


def check_vanilla_vae():
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


if __name__ == '__main__':
    check_hf_vae()
