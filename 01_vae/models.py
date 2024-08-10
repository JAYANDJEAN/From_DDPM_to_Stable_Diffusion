from abc import abstractmethod
import torch
from torch.nn import functional as F
from torch import nn, Tensor
from typing import List, Any


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: str) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VanillaVAE(BaseVAE):
    def __init__(self, in_channels: int, image_size: int, latent_dim: int, hidden_dims: List = None) -> None:
        super(VanillaVAE, self).__init__()
        assert image_size % 32 == 0
        self.scale = image_size // 32
        self.latent_dim = latent_dim

        if hidden_dims is None:
            self.hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        modules = []
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(self.hidden_dims[-1] * self.scale * self.scale, latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1] * self.scale * self.scale, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.hidden_dims[-1] * self.scale * self.scale)

        self.hidden_dims.reverse()
        modules = []
        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i], self.hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[-1], self.hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(self.hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, x: Tensor) -> List[Tensor]:
        # x: (batch_size, n_channel, width, height)
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        # mu, log_var: (batch_size, latent_dim)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[0], self.scale, self.scale)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    @staticmethod
    def reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        # return (batch_size, latent_dim)
        return eps * std + mu

    def forward(self, x: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        x = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self, num_samples: int, device: str) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(device)
        samples = self.decode(z)
        return samples


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, n_channels: int):
        super().__init__()
        self.in_proj = nn.Linear(n_channels, 3 * n_channels, bias=False)
        self.out_proj = nn.Linear(n_channels, n_channels)
        self.n_heads = n_heads
        self.dim_head = n_channels // n_heads

    def forward(self, x: Tensor) -> Tensor:
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.dim_head)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(interim_shape).transpose(1, 2), (q, k, v))

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(input_shape)
        out = self.out_proj(out)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: Tensor):
        residue = x
        x = self.norm(x)
        n, c, h, w = x.shape
        x = x.view((n, c, h * w)).transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2).view((n, c, h, w))
        x += residue
        return x


class ResidualBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        assert ch_in % 32 == 0
        self.conv_1 = nn.Sequential(
            nn.GroupNorm(32, ch_in),
            nn.SiLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        )
        self.conv_2 = nn.Sequential(
            nn.GroupNorm(32, ch_out),
            nn.SiLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        )
        if ch_in != ch_out:
            self.residual_layer = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, x: Tensor):
        out = self.conv_2(self.conv_1(x)) + self.residual_layer(x)
        return out


# todo
class SDVAE(BaseVAE):
    def __init__(self, in_channels: int, image_size: int, latent_dim: int, hidden_dims: List = None):
        super().__init__()
        assert image_size % 8 == 0
        self.scale = image_size // 8
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        assert len(hidden_dims) == 4

        modules = []
        for i in range(len(hidden_dims)):
            ch_in = in_channels if i == 0 else hidden_dims[i - 1]
            stride = 1 if i == 0 else 2
            ch_out = hidden_dims[i]
            modules.append(
                nn.Sequential(
                    nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1),
                    ResidualBlock(ch_out, ch_out)
                )
            )
        modules.append(
            nn.Sequential(
                AttentionBlock(hidden_dims[-1]),
                ResidualBlock(hidden_dims[-1], hidden_dims[-1]),
                nn.GroupNorm(32, hidden_dims[-1]),
                nn.SiLU(),
                nn.Conv2d(hidden_dims[-1], 8, kernel_size=3, stride=1, padding=1),
            )
        )
        self.encoder = nn.Sequential(*modules)

        hidden_dims.reverse()

        modules = [
            nn.Sequential(
                nn.Conv2d(4, hidden_dims[0], kernel_size=3, stride=1, padding=1),
                ResidualBlock(hidden_dims[0], hidden_dims[0]),
                AttentionBlock(hidden_dims[0]),
            )
        ]

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=1, padding=1),
                    ResidualBlock(hidden_dims[i + 1], hidden_dims[i + 1]),
                )
            )
        modules.append(
            nn.Sequential(
                nn.GroupNorm(8, hidden_dims[-1]),
                nn.SiLU(),
                nn.Conv2d(hidden_dims[-1], in_channels, kernel_size=3, stride=1, padding=1),
            )
        )

        self.decoder = nn.Sequential(*modules)

    def encode(self, x: Tensor) -> Tensor:
        # x: (bs, 3, 512, 512) -> (bs, 4, 64, 64)
        x = self.encoder(x)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_variance, -30, 20)
        std = log_variance.exp().sqrt()

        noise = torch.randn_like(std)
        x = mean + std * noise
        return x

    def decode(self, z: Tensor) -> Tensor:
        # x: (bs, 4, 64, 64) -> (bs, 3, 512, 512)
        return self.decoder(z)
