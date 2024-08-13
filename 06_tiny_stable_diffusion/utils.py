import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Optional

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]


def denormalize(tensor):
    device = tensor.device
    mean = torch.tensor(means).view(1, 3, 1, 1).to(device)
    std = torch.tensor(stds).view(1, 3, 1, 1).to(device)
    return tensor * std + mean


def animal_faces_loader(batch_size: int, img_size: Optional[int]):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])
    dataset = datasets.ImageFolder(root='../00_assets/datasets/afhq/train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化 shadow 权重
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """使用当前模型的参数更新 shadow 权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """保存当前模型参数，并将模型参数更新为 shadow 权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """恢复模型参数为原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


class CosineWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_lr, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_epochs - warmup_epochs))
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr + (self.max_lr - base_lr) * self.last_epoch / self.warmup_epochs
                    for base_lr in self.base_lrs]
        else:
            return self.cosine_scheduler.get_lr()

    def step(self, epoch=None, metrics=None):
        if self.last_epoch >= self.warmup_epochs:
            self.cosine_scheduler.step(epoch)
        else:
            return super().step(epoch)


class TrainerDDPM(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        """
        \alpha_t:=1-\beta_t
        \bar{\alpha}_t:=\prod_{s=1}^t \alpha_s
        """
        super().__init__()
        self.model = model
        self.T = T
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, labels):
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        # extract 计算第t步加了噪音的图片，noisy_img
        x_t = (extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
               extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        pred_noise = self.model(x_t, t, labels)
        loss = F.mse_loss(pred_noise, noise, reduction='none')
        return loss


class SamplerDDPM(nn.Module):
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
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t=x_t, t=t, labels=labels)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
            # img_save = torch.clip(x_t * 0.5 + 0.5, 0, 1)
            # save_image(img_save, os.path.join(self.save_path, f'{self.T - time_step:04d}.png'), nrow=self.nrow)
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
