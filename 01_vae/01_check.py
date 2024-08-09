import torch
from models import VanillaVAE


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


check_vae()
