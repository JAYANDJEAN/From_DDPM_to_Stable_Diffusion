import os
from typing import Dict
import numpy as np

import torch
import torch.optim as optim
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets, transforms

from utils import SamplerDDPM, TrainerDDPM, GradualWarmupScheduler, denormalize
from unet import UNet


def train(config: Dict):
    print("Model Training...............")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"use device: {device}")

    transform = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])
    dataset = datasets.ImageFolder(root='../00_assets/datasets/afhq/train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    _, batch = next(enumerate(dataloader))
    img_batch = torch.clip(denormalize(batch[0].clone(), means, stds, device), 0, 1)
    save_image(img_batch, "../00_assets/image/animal_faces.png", nrow=config["nrow"])

    net_model = UNet(channel_img=config["img_channel"], channel_base=config["channel"],
                     channel_mults=config["channel_mult"], num_res_blocks=config["num_res_blocks"],
                     n_steps=config["T"], dropout=config["dropout"]).to(device)
    print("Total trainable parameters:", sum(p.numel() for p in net_model.parameters() if p.requires_grad))
    epoch_awoken = config["epoch_awoken"]
    if epoch_awoken is not None:
        assert isinstance(epoch_awoken, int)
        net_model.load_state_dict(torch.load(
            os.path.join(config["model_dir"], f"ckpt_{epoch_awoken}.pth"),
            map_location=device), strict=False)
        print(f"Model weight has loaded from ckpt_{epoch_awoken}.pth!")
        base = epoch_awoken
    else:
        base = 0

    optimizer = torch.optim.AdamW(net_model.parameters(), lr=config["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=config["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=config["multiplier"],
                                             warm_epoch=config["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = TrainerDDPM(net_model, config["beta_1"], config["beta_T"], config["T"]).to(device)

    for epoch in range(config["epoch"]):
        start_time = timer()
        losses = 0
        for images, labels in dataloader:
            optimizer.zero_grad()
            x_0 = images.to(device)
            labels = labels.to(device) + 1  # why
            if np.random.rand() < 0.1:
                labels = torch.zeros_like(labels).to(device)
            loss = trainer(x_0, labels).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), config["grad_clip"])
            optimizer.step()
            losses += loss.item()
        train_loss = losses / len(list(dataloader))
        end_time = timer()
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")

        warmUpScheduler.step()

        if epoch > config["epoch_save"]:
            torch.save(net_model.state_dict(), os.path.join(config["model_dir"], f"ckpt_{base + epoch}.pth"))


def generate(config: Dict):
    print("Images Generating...............")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load model and evaluate
    with torch.no_grad():
        values = torch.arange(1, config["num_class"] + 1)
        labels = values.repeat_interleave(config["nrow"]).to(device)

        model = UNet(channel_img=config["img_channel"], channel_base=config["channel"],
                     channel_mults=config["channel_mult"], n_steps=config["T"],
                     num_res_blocks=config["num_res_blocks"], dropout=config["dropout"]).to(device)
        base = config["epoch_awoken"] if config["epoch_awoken"] is not None else 0

        for i in range(config["epoch_save"] + base, config["epoch"] + base):
            ckpt = torch.load(os.path.join(config["model_dir"], f"ckpt_{i}.pth"), map_location=device)
            model.load_state_dict(ckpt)
            model.eval()
            sampler = SamplerDDPM(
                model, config["beta_1"], config["beta_T"], config["T"], w=config["w"]).to(device)

            img_noisy = torch.randn(size=[config["num_class"] * config["nrow"], config["img_channel"],
                                          config["img_size"], config["img_size"]], device=device)
            img_sample = sampler(img_noisy, labels)
            save_image(tensor=denormalize(img_sample, means, stds, device),
                       fp=f"../00_assets/image/animal_face_generated_{i}.png",
                       nrow=config["nrow"])
            print(f"animal_face_generated_{i}.png is done!")


if __name__ == '__main__':
    modelConfig = {
        "epoch": 40,
        "epoch_save": 30,
        "epoch_awoken": None,
        "batch_size": 32,
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_channel": 3,
        "img_size": 64,
        "grad_clip": 1.,
        "w": 1.8,
        "nrow": 8,
        "num_class": 3,
        "model_dir": "../00_assets/model_animal3/"
    }

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    os.makedirs(modelConfig["model_dir"], exist_ok=True)

    train(modelConfig)
    generate(modelConfig)
