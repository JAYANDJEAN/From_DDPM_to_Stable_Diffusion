import torch
from unet import UNet
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda
from utils import *
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets import CIFAR10
from schedulers import *
from utils import *
from unet import UNet
import yaml
from torch import nn
from torch.optim import Adam

batch_size = 128
n_step = 1000
n_class = 10
n_time_embed = 100

torch.manual_seed(0)
with open('../00_assets/cifar.yaml', 'r') as file:
    config = yaml.safe_load(file)


class DDPM(nn.Module):
    def __init__(self,
                 network: nn.Module,
                 n_steps: int,
                 min_beta: float,
                 max_beta: float,
                 device):
        super().__init__()
        self.network = network
        self.device = device
        sch = LinearScheduler(n_steps, min_beta, max_beta)
        self.alphas_hat = sch.alphas_hat.to(device)
        self.alphas = sch.alphas.to(device)
        self.betas = sch.betas.to(device)

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        return self.network(x, t, y)

    def noisy_(self, x: Tensor, t: Tensor, eta: Tensor) -> Tensor:
        return xt_from_x0(self.alphas_hat, x, t, eta)


def check_data():
    transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
    dataset = CIFAR10(root="../00_assets/datasets", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in loader:
        show_images(batch[0], "../00_assets/CIFAR_BATCH")
        break

    unet = UNet(channels=[3, 8, 16, 32, 64, 64], time_emb_dim=100, num_class=config['num_class'])
    ddpm = DDPM(unet, config['n_steps'], config['min_b'], config['max_b'], config['device']).to(config['device'])
    ddpm.load_state_dict(torch.load(config['model_path'], map_location="cpu"))
    ddpm.eval()

    print("\nGenerating images.......")
    generated = generate_new_images(ddpm=ddpm, config=config)
    show_images(generated, f"../00_assets/ddpm_{config['dt']}_class_{config['label']}")


def check_unet_output():
    t = torch.randint(0, n_step, (batch_size,))
    y = torch.randint(0, n_class, (batch_size,))

    x0 = torch.rand(batch_size, 1, 28, 28)
    unet = UNet(channels=[1, 10, 20, 40, 64],
                dropout=0.1,
                time_emb_dim=n_time_embed,
                n_steps=n_step,
                num_class=n_class)
    x_recon = unet(x0, t, None)
    assert x_recon.shape == x0.shape

    x1 = torch.rand(batch_size, 3, 32, 35)
    unet = UNet(channels=[3, 32, 64, 128],
                dropout=0.1,
                time_emb_dim=100,
                n_steps=n_step,
                num_class=n_class)
    x_recon = unet(x1, t, y)
    assert x_recon.shape == x1.shape


if __name__ == '__main__':
    check_data()
