import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import imageio
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import LRScheduler


class GradualWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = None
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, labels):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
              extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        loss = F.mse_loss(self.model(x_t, t, labels), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w=0.):
        super().__init__()

        self.model = model
        self.T = T
        """
        In the classifier free guidance paper, w is the key to control the guidance.
        w = 0 and with label = 0 means no guidance.
        w > 0 and label > 0 means guidance. Guidance would be stronger if w is bigger.
        """

        self.w = w

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def p_mean_variance(self, x_t, t, labels):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = self.model(x_t, t, labels)
        nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))
        eps = (1. + self.w) * eps - self.w * nonEps
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T, labels):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t=x_t, t=t, labels=labels)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)


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


def generate_new_images(ddpm, config, n_samples=100, frames_per_gif=10):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
    frames = []
    if config['dt'] == 'mnist':
        c, h, w = 1, 28, 28
    elif config['dt'] == 'cifar10':
        c, h, w = 3, 32, 32
    else:
        raise ValueError("Unsupported dataset type")

    with torch.no_grad():
        x = torch.randn(n_samples, c, h, w).to(config['device'])
        for idx, t in tqdm(enumerate(list(range(config['n_steps']))[::-1])):
            t_tensor = torch.as_tensor([t] * n_samples).to(config['device'])
            if config['with_class']:
                y_tensor = torch.as_tensor([config['label']] * n_samples).to(config['device'])
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

            if idx % frames_per_gif == 0:
                normalized = (x - x.min()) / (x.max() - x.min()) * 255
                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                frame = frame.cpu().numpy().astype(np.uint8)
                frames.append(frame)

    gif_path = f"../00_assets/ddpm_{config['dt']}_class_{config['label']}.gif"
    with imageio.get_writer(gif_path, mode="I") as writer:
        for idx, frame in enumerate(frames):
            rgb_frame = np.repeat(frame, 3, axis=2)
            writer.append_data(rgb_frame)

            if idx == len(frames) - 1:
                last_rgb_frame = np.repeat(frames[-1], 3, axis=2)
                for _ in range(frames_per_gif // 3):
                    writer.append_data(last_rgb_frame)

    return x
