import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
import imageio
import einops





def show_images(images, title="sample"):
    """Shows the provided images as sub-pictures in a square"""

    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    fig = plt.figure(figsize=(10, 10))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < len(images):
                ax = fig.add_subplot(rows, cols, idx + 1)
                image = images[idx].transpose((1, 2, 0))
                image = (image / 2.0) + 0.5
                ax.imshow(image)
                ax.axis('off')
                idx += 1
    fig.suptitle(title, fontsize=30)
    plt.savefig(title + '.png')


def show_first_batch(loader):
    for batch in loader:
        show_images(batch[0], "Images in the first batch")
        break


def generate_new_images(ddpm,
                        n_samples=100,
                        device=None,
                        n_steps=1000,
                        time_embed_size=100,
                        frames_per_gif=100,
                        gif_name="sampling.gif",
                        c=1, h=28, w=28,
                        with_class=False,
                        num_classes=10,
                        label=None):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
    frame_idxs = np.linspace(0, n_steps, frames_per_gif).astype(np.uint)
    frames = []
    time_embedding = nn.Embedding(n_steps, time_embed_size)
    time_embedding.weight.data = sinusoidal_embedding(n_steps, time_embed_size)
    time_embedding.requires_grad_(False)

    with torch.no_grad():
        if device is None:
            device = ddpm.device
        x = torch.randn(n_samples, c, h, w).to(device)

        for idx, t in tqdm(enumerate(list(range(n_steps))[::-1])):
            time_embed = time_embedding(torch.ones(n_samples, ).int() * t).squeeze().to(device)
            if with_class:
                class_embed = torch.zeros(n_samples, num_classes).to(device)
                class_embed[:, label] = 1
                eta_theta = ddpm(x, time_embed, class_embed)
            else:
                eta_theta = ddpm(x, time_embed, None)

            alpha_t = ddpm.alphas[t]
            alpha_t_hat = ddpm.alphas_hat[t]

            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_hat).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()
                x = x + sigma_t * z

            if idx in frame_idxs or t == 0:
                # Putting digits in range [0, 255]
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])
                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                frame = frame.cpu().numpy().astype(np.uint8)
                frames.append(frame)

    # Storing the gif
    with imageio.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):
            rgb_frame = np.repeat(frame, 3, axis=2)
            writer.append_data(rgb_frame)

            # Showing the last frame for a longer time
            if idx == len(frames) - 1:
                last_rgb_frame = np.repeat(frames[-1], 3, axis=2)
                for _ in range(frames_per_gif // 3):
                    writer.append_data(last_rgb_frame)
    return x
