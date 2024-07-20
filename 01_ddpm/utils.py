import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm

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


def generate_new_images(ddpm, config,
                        n_samples=100,
                        frames_per_gif=100):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
    frame_idxs = np.linspace(0, config['n_steps'], frames_per_gif).astype(np.uint)
    frames = []
    if config['dt'] == 'mnist':
        c, h, w = 1, 28, 28
    elif config['dt'] == 'cifar10':
        c, h, w = 3, 32, 32

    with torch.no_grad():
        x = torch.randn(n_samples, c, h, w).to(config['device'])
        for idx, t in tqdm(enumerate(list(range(config['n_steps']))[::-1])):
            if config['with_class']:
                y_tensor = torch.as_tensor([config['label']] * n_samples).to(config['device'])
                t_tensor = torch.as_tensor([t] * n_samples).to(config['device'])
                eta_theta = ddpm(x, t_tensor, y_tensor)
            else:
                eta_theta = ddpm(x, t_tensor, None)
            alpha_t = ddpm.alphas[t]
            alpha_t_hat = ddpm.alphas_hat[t]
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_hat).sqrt() * eta_theta)
            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(config['device'])
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
    with imageio.get_writer(f"../00_assets/ddpm_{config['dt']}_class_{config['label']}.gif", mode="I") as writer:
        for idx, frame in enumerate(frames):
            rgb_frame = np.repeat(frame, 3, axis=2)
            writer.append_data(rgb_frame)

            # Showing the last frame for a longer time
            if idx == len(frames) - 1:
                last_rgb_frame = np.repeat(frames[-1], 3, axis=2)
                for _ in range(frames_per_gif // 3):
                    writer.append_data(last_rgb_frame)
    return x
