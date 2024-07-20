from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST

from schedulers import *
from utils import *
from unet import UNet

torch.manual_seed(0)
all_data = True
partial = 7
n_step, min_b, max_b, n_class, n_time, batch_size, n_epoch, lr = (1000, 10 ** -4, 0.02, 10, 100, 128, 20, 0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
store_path = "ddpm_model_with_class.pth"
label = 8
result_name = f"ddpm_result_with_class_{label}"
gif_name = f"ddpm_with_class_{label}.gif"

# Data
transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
full_dataset = MNIST(root="./datasets", train=True, transform=transform, download=True)
if all_data:
    loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
else:
    subset_indices = [i for i, (img, label) in enumerate(full_dataset) if label == partial]
    subset_dataset = Subset(full_dataset, subset_indices)
    loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)


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
            num_class=n_class)
ddpm = DDPM(unet, n_step, min_b, max_b, device).to(device)
print(f"\nnumber of parameters: {sum([p.numel() for p in ddpm.parameters()])}")

print("\nModel training.......")
training_loop(ddpm, loader, device, lr, n_epoch, n_step, store_path, True)

print("\nModel loading.......")
best_model = DDPM(unet, n_step, min_b, max_b, device).to(device)
best_model.load_state_dict(torch.load(store_path, map_location=device))
best_model.eval()

print("\nGenerating images.......")
generated = generate_new_images(ddpm=best_model, n_steps=n_step, gif_name=gif_name,
                                time_embed_size=n_time, device=device,
                                with_class=True, num_classes=n_class, label=label)
show_images(generated, result_name)
