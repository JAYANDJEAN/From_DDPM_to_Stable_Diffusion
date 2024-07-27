from stable_diffusion_pytorch import tokenizer, model_loader, util
from stable_diffusion_pytorch.samplers import KLMSSampler
from stable_diffusion_pytorch.diffusion import AttentionBlock, ResidualBlock, TimeEmbedding, UNet
import torch
from tqdm import tqdm
from PIL import Image
from modelsummary import summary
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prompts = ["a photograph of an astronaut riding a horse"]
height = width = 512
cfg_scale = 7.5
n_inference_steps = 50
tokenizer = tokenizer.Tokenizer()


def check_pipeline():
    with torch.no_grad():
        clip = model_loader.load_clip(device)
        clip.to(device)
        uncond_prompts = [""] * len(prompts)
        uncond_tokens = torch.tensor(tokenizer.encode_batch(uncond_prompts), dtype=torch.long, device=device)
        uncond_context = clip(uncond_tokens)
        cond_tokens = torch.as_tensor(tokenizer.encode_batch(prompts), dtype=torch.long, device=device)
        # torch.Size([2, 77]) (num_prompt, token_max_length)
        print(f"tokens shape: {cond_tokens.shape}")
        cond_context = clip(cond_tokens)
        # torch.Size([2, 77, 768]) (num_prompt, token_max_length, dim_embedding)
        context = torch.cat([cond_context, uncond_context])
        print(f"context shape: {context.shape}")
        del clip

        # 它管理时间步长、噪声调整和更新，并确保生成过程的稳定性和效果。
        sampler = KLMSSampler(n_inference_steps=n_inference_steps)
        # 在这种潜在空间中，4 个通道的表示方式是比较常见的。这些通道并不直接对应于 RGB 图像的颜色通道，而是更高维的特征表示
        noise_shape = (len(prompts), 4, height // 8, width // 8)
        latents = torch.randn(noise_shape, device=device, dtype=torch.float32)
        latents *= sampler.initial_scale
        print(f"sample initial: {sampler.initial_scale}")

        timesteps = tqdm(sampler.timesteps)
        diffusion = model_loader.load_diffusion(device)
        diffusion.to(device)
        for i, timestep in enumerate(timesteps):
            time_embedding = util.get_time_embedding(timestep, torch.float32).to(device)
            # torch.Size([1, 320])
            print(f"time_embedding shape: {time_embedding.shape}")
            input_latents = latents * sampler.get_input_scale()
            # (batch_size, channels, height, width) -> (2 * batch_size, channels, height, width)
            # 在扩散模型中，这样的操作通常用于同时计算条件和无条件的输出，以便进行条件指导（Conditional Guidance）
            input_latents = input_latents.repeat(2, 1, 1, 1)
            output = diffusion(input_latents, context, time_embedding)
            # 分割输出为条件和无条件部分
            output_cond, output_uncond = output.chunk(2)
            output = cfg_scale * (output_cond - output_uncond) + output_uncond
            latents = sampler.step(latents, output)
            # torch.Size([1, 4, 64, 64])
            print(f"latents shape: {latents.shape}")
        del diffusion

        decoder = model_loader.load_decoder(device)
        decoder.to(device)
        images = decoder(latents)
        # torch.Size([1, 3, 512, 512])
        print(f"images shape: {images.shape}")
        del decoder

        images = util.rescale(images, (-1, 1), (0, 255), clamp=True)
        images = util.move_channel(images, to="last")
        # torch.Size([1, 512, 512, 3])
        print(f"images shape: {images.shape}")
        images = images.to('cpu', torch.uint8).numpy()
        images = [Image.fromarray(image) for image in images]
        images[0].save('output.jpg')


def check_diffusion():
    latent = torch.randn((1, 4, 64, 64), dtype=torch.float32)
    context = torch.randn((2, 77, 768), dtype=torch.float32)
    time_embedding = torch.randn((1, 320), dtype=torch.float32)
    te = TimeEmbedding(320)
    time = te(time_embedding)

    unet = UNet()
    output = unet(latent, context, time)
    print(output.shape)

    diffusion = model_loader.load_diffusion(device)
    summary(diffusion, latent, context, time_embedding, show_input=True)


if __name__ == '__main__':
    check_diffusion()
