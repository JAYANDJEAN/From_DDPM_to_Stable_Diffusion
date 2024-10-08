import torch
from torchvision import transforms
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from models import VanillaVAE, VQVAE
from utils import animal_faces_loader, denormalize


def check_animal_faces():
    dataloader = animal_faces_loader('train', 14, 128)
    for images, labels in dataloader:
        print(images.shape)
        save_image(tensor=denormalize(images.clone()),
                   fp=f"../00_assets/image/animal_faces_generated_method1.png",
                   nrow=7,
                   padding=0)
        break


def check_hf_vae():
    """
    对比了一下，stabilityai/sdxl-vae貌似更好
    """
    sdxl_vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    sdxl_vae_fix = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
    batch_size = 7
    dataloader = animal_faces_loader('train', batch_size, 512)
    _, batch = next(enumerate(dataloader))

    with torch.no_grad():
        latents = sdxl_vae.encode(batch[0]).latent_dist.sample()
        print(latents.shape)
        decoded_image = sdxl_vae.decode(latents).sample
        print(decoded_image.shape)

        result = torch.cat((batch[0], decoded_image), dim=0)
        result = transforms.Resize((128, 128))(result)
        save_image(tensor=denormalize(result),
                   fp=f'../00_assets/image/animal_faces_sdxl_vae.png',
                   nrow=batch_size,
                   padding=0)


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


def check_vqvae():
    batch_size = 7
    vqvae = VQVAE(in_channels=3, img_size=512, embedding_dim=4, num_embeddings=128, hidden_dims=[32, 64, 128])

    x = torch.randn((2, 3, 512, 512))
    latent = vqvae.encode(x)
    print(latent[0].shape)


if __name__ == '__main__':
    check_vqvae()
