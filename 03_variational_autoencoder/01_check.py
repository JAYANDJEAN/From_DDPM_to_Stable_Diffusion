import torch
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
from models import VanillaVAE, VQVAE
from utils import denormalize, animal_faces_loader


def check_hf_vae():
    """
    对比了一下，stabilityai/sdxl-vae貌似更好
    """
    sdxl_vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    sdxl_vae_fix = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")

    dataloader = animal_faces_loader(img_size=512, batch_size=8)
    _, batch = next(enumerate(dataloader))
    save_image(tensor=denormalize(batch[0].clone()),
               fp=f"../00_assets/image/animal_faces_raw.png",
               nrow=4)
    with torch.no_grad():
        latents = sdxl_vae.encode(batch[0]).latent_dist.sample()
        print(latents.shape)
        decoded_image = sdxl_vae.decode(latents).sample
        print(decoded_image.shape)
        save_image(tensor=denormalize(decoded_image.clone()),
                   fp=f"../00_assets/image/animal_faces_sdxl_vae_latent.png",
                   nrow=4)
        latents = sdxl_vae_fix.encode(batch[0]).latent_dist.sample()
        print(latents.shape)
        decoded_image = sdxl_vae_fix.decode(latents).sample
        print(decoded_image.shape)
        save_image(tensor=denormalize(decoded_image.clone()),
                   fp=f"../00_assets/image/animal_faces_sdxl_vae_fix_latent.png",
                   nrow=4)


def check_vae():
    x = torch.randn((128, 3, 128, 128))
    vae = VanillaVAE(in_channels=3, image_size=128, latent_dim=256)
    z, mu, log_var = vae.encode(x)
    print(f"mu shape: {mu.shape}, var shape: {log_var.shape}, z shape: {z.shape}")
    recon = vae.decode(z)
    print(f"recons shape: {recon.shape}")
    result = vae(x)
    loss = vae.loss_function(*result, M_N=0.005)
    print(loss)

    print("=" * 70)
    vqvae = VQVAE(in_channels=3, embedding_dim=4, num_embeddings=3, img_size=512)
    x = torch.randn(16, 3, 512, 512)
    latent = vqvae.encode(x)
    print("Model encoded size:", latent[0].size())
    recon = vqvae.decode(latent[0])
    print(f"recons shape: {recon.shape}")

    result = vqvae(x)
    loss = vqvae.loss_function(*result, M_N=0.005)
    print(loss)


if __name__ == '__main__':
    check_vae()
