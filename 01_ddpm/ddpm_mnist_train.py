from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST

from schedulers import *
from utils import *
from unet import UNet
import yaml

torch.manual_seed(0)
with open('../00_assets/ddpm_mnist.yml', 'r') as file:
    config = yaml.safe_load(file)

label = 8
result_name = f"ddpm_result_with_class_{label}"
gif_name = f"ddpm_with_class_{label}.gif"

# Data
transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
full_dataset = MNIST(root="../00_assets/datasets", train=True, transform=transform, download=True)
loader = DataLoader(full_dataset, batch_size=config['batch_size'], shuffle=True)


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


channels = [1, 8, 16, 32, 64, 64]
unet = UNet(channels=channels,
            time_emb_dim=100,
            num_class=config['num_class'])
ddpm = DDPM(unet, config['n_step'], config['min_b'], config['max_b'], config['device']).to(config['device'])
print(f"\nnumber of parameters: {sum([p.numel() for p in ddpm.parameters()])}")

print("\nModel training.......")


def training_loop():
    mse = nn.MSELoss()
    best_loss = float("inf")
    optim = Adam(ddpm.parameters(), lr)

    for epoch in tqdm(range(n_epochs), desc=f"Training progress"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}")):
            # Loading data
            x0 = batch[0].to(device)
            y0 = batch[1].to(device)
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (len(x0),)).to(device)
            noisy_img = ddpm.noisy_(x0, t, eta).to(device)

            if with_class:
                pred_eta = ddpm(noisy_img, t, y0).to(device)
            else:
                pred_eta = ddpm(noisy_img, t, None).to(device)
            loss = mse(pred_eta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * len(x0) / len(loader.dataset)
        log_string = f"\nLoss at epoch {epoch + 1}: {epoch_loss:.3f}"

        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"
        print(log_string)




print("\nModel loading.......")
best_model = DDPM(unet, config['n_step'], config['min_b'], config['max_b'], device).to(device)
best_model.load_state_dict(torch.load(store_path, map_location=device))
best_model.eval()

print("\nGenerating images.......")
generated = generate_new_images(ddpm=best_model, n_steps=n_step, gif_name=gif_name,
                                time_embed_size=n_time, device=device,
                                with_class=True, num_classes=n_class, label=label)
show_images(generated, result_name)
