from abc import abstractmethod
import torch
from torch.nn import functional as F
from torch import nn, Tensor
from typing import List, Any


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, x: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, z: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: str, **kwargs) -> Tensor:
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

        # mu, log_var: (batch_size, latent_dim)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        # Re-parameterization trick to sample from N(mu, var) from N(0,1).
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        # z: (batch_size, latent_dim)
        z = eps * std + mu
        return [z, mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[0], self.scale, self.scale)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, x: Tensor, **kwargs) -> List[Tensor]:
        z, mu, log_var = self.encode(x)
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

    def sample(self, num_samples: int, device: str, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(device)
        samples = self.decode(z)
        return samples


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> tuple[Tensor, Tensor]:
        # [B x D x H x W] -> [B x H x W x D]
        latents = latents.permute(0, 2, 3, 1).contiguous()
        latents_shape = latents.shape
        # [BHW x D]
        flat_latents = latents.view(-1, self.D)

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_index = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_index.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_index, 1)  # [BHW x K]
        print(encoding_one_hot)

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]
        print(quantized_latents)

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        quantized_latents = quantized_latents.permute(0, 3, 1, 2).contiguous()
        return quantized_latents, vq_loss  # [B x D x H x W]


class ResidualLayer(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0, bias=False)
        )
        if ch_in != ch_out:
            self.residual = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        else:
            self.residual = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x) + self.residual(x)


# copy from sd1 vae
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


# copy from sd1 vae
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


# copy from sd1 vae
class ResidualBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        assert ch_in % 32 == 0
        self.conv = nn.Sequential(
            nn.GroupNorm(32, ch_in),
            nn.SiLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, ch_out),
            nn.SiLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        )
        if ch_in != ch_out:
            self.residual = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        else:
            self.residual = nn.Identity()

    def forward(self, x: Tensor):
        out = self.conv(x) + self.residual(x)
        return out


class VQVAE(BaseVAE):
    def __init__(self, in_channels: int, embedding_dim: int, num_embeddings: int,
                 hidden_dims: List = None, beta: float = 0.25, img_size: int = 64, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU()
            )
        )

        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1),
                nn.LeakyReLU()
            )
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, self.beta)

        # Build Decoder
        modules = [
            nn.Sequential(
                nn.Conv2d(embedding_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU()
            )
        ]

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))
        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU()
                )
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], out_channels=3, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            )
        )

        self.decoder = nn.Sequential(*modules)

    def encode(self, x: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param x: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(x)
        return [result]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(z)
        return result

    def forward(self, x: Tensor, **kwargs) -> List[Tensor]:
        encoding = self.encode(x)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), x, vq_loss]

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        x = args[1]
        vq_loss = args[2]
        recons_loss = F.mse_loss(recons, x)
        loss = recons_loss + vq_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'VQ_Loss': vq_loss}

    def sample(self, num_samples: int, current_device: str, **kwargs) -> Tensor:
        raise Warning('VQVAE sampler is not implemented.')
