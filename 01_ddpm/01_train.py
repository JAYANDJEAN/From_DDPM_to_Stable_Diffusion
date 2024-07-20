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

torch.manual_seed(0)
with open('../00_assets/ddpm.yml', 'r') as file:
    config = yaml.safe_load(file)


def training_loop():
    mse = nn.MSELoss()
    best_loss = float("inf")
    optim = Adam(ddpm.parameters(), config['lr'])

    for epoch in tqdm(range(config['n_epochs']), desc=f"Training progress"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{config['n_epochs']}")):
            # Loading data
            device = config['device']
            x0 = batch[0].to(device)
            y0 = batch[1].to(device)
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, config['n_steps'], (len(x0),)).to(device)
            noisy_img = ddpm.noisy_(x0, t, eta).to(device)

            if config['with_class']:
                pred_eta = ddpm(noisy_img, t, y0).to(device)
            else:
                pred_eta = ddpm(noisy_img, t, None).to(device)
            loss = mse(pred_eta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * len(x0) / len(loader)
        log_string = f"\nLoss at epoch {epoch + 1}: {epoch_loss:.3f}"

        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), config['model_path'])
            log_string += " --> Best model ever (stored)"
        print(log_string)


transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
if config['dt'] == 'mnist':
    dataset = MNIST(root="../00_assets/datasets", train=True, transform=transform, download=True)
    channels = [1, 8, 16, 32, 64, 64]
elif config['dt'] == 'cifar10':
    dataset = CIFAR10(root="../00_assets/datasets", train=True, download=True, transform=transform)
    channels = [3, 8, 16, 32, 64, 64]
else:
    dataset = None
    channels = None
loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
show_first_batch(loader)


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


unet = UNet(channels=channels,
            time_emb_dim=100,
            num_class=config['num_class'])
ddpm = DDPM(unet, config['n_steps'], config['min_b'], config['max_b'], config['device']).to(config['device'])
print(f"\nnumber of parameters: {sum([p.numel() for p in ddpm.parameters()])}")

print("\nModel training.......")
training_loop()

print("\nModel loading.......")
best_model = DDPM(unet, config['n_steps'], config['min_b'], config['max_b'], config['device']).to(config['device'])
best_model.load_state_dict(torch.load(config['model_path']))
best_model.eval()

print("\nGenerating images.......")
generated = generate_new_images(ddpm=best_model, config=config)
show_images(generated, f"../00_assets/ddpm_{config['dt']}_class_{config['label']}")
