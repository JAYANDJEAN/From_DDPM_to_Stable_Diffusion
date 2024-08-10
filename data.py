import torch
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets, transforms
from typing import Optional

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]


def denormalize(tensor):
    device = tensor.device
    mean = torch.tensor(means).view(1, 3, 1, 1).to(device)
    std = torch.tensor(stds).view(1, 3, 1, 1).to(device)
    tensor = tensor
    return tensor * std + mean


def animal_faces_loader(batch_size: int, img_size: Optional[int]):
    transform = transforms.Compose([
        transforms.Resize(img_size, img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])
    dataset = datasets.ImageFolder(root='./00_assets/datasets/afhq/train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader
