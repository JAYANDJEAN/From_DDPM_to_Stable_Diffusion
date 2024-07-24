import os
from typing import Dict
import numpy as np
import glob
from PIL import Image

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from utils import GaussianDiffusionSampler, GaussianDiffusionTrainer, GradualWarmupScheduler
from unet import UNet


def train(config: Dict):
    print("Model Training...............")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = CIFAR10(root='../00_assets/datasets', train=True, download=True,
                      transform=Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]))
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True, pin_memory=True)
    _, batch = next(enumerate(dataloader))
    img_batch = torch.clip(batch[0] * 0.5 + 0.5, 0, 1)
    save_image(img_batch, os.path.join(config["image_dir"], config["raw_name"]), nrow=config["nrow"])

    net_model = UNet(channel_img=config["img_channel"], channel_base=config["channel"],
                     channel_mults=config["channel_mult"], num_res_blocks=config["num_res_blocks"],
                     n_steps=config["T"], dropout=config["dropout"]).to(device)
    print("Total trainable parameters:", sum(p.numel() for p in net_model.parameters() if p.requires_grad))
    if config["training_weight"] is not None:
        net_model.load_state_dict(torch.load(
            os.path.join(config["model_dir"], config["training_weight"]), map_location=device), strict=False)
        print("Model weight load down.")

    optimizer = torch.optim.AdamW(net_model.parameters(), lr=config["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=config["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=config["multiplier"],
                                             warm_epoch=config["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(net_model, config["beta_1"], config["beta_T"], config["T"]).to(device)

    for e in range(config["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                optimizer.zero_grad()
                b = images.shape[0]
                x_0 = images.to(device)
                labels = labels.to(device) + 1  # why
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)
                loss = trainer(x_0, labels).sum() / b ** 2.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), config["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()

        torch.save(net_model.state_dict(), os.path.join(config["model_dir"], 'ckpt_' + str(e) + ".pth"))


def generate(config: Dict):
    print("Images Generating...............")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load model and evaluate
    with torch.no_grad():
        values = torch.arange(1, config["num_class"] + 1)
        labels = values.repeat_interleave(config["nrow"]).to(device)
        print("labels: ", labels)
        model = UNet(channel_img=3, channel_base=config["channel"], channel_mults=config["channel_mult"],
                     num_res_blocks=config["num_res_blocks"], dropout=config["dropout"]).to(device)
        ckpt = torch.load(os.path.join(config["model_dir"], config["eval_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, config["beta_1"], config["beta_T"], config["T"],
            save_path=config["image_gen_dir"],
            nrow=config["nrow"],
            w=config["w"]).to(device)

        img_noisy = torch.randn(size=[config["num_class"] * config["nrow"], config["img_channel"],
                                      config["img_size"], config["img_size"]], device=device)
        save_image(tensor=torch.clip(img_noisy * 0.5 + 0.5, 0, 1),
                   fp=os.path.join(config["image_dir"], config["noisy_name"]),
                   nrow=config["nrow"])
        img_sample = sampler(img_noisy, labels)
        save_image(tensor=img_sample * 0.5 + 0.5,
                   fp=os.path.join(config["image_dir"], config["generate_name"]),
                   nrow=config["nrow"])

    file_path = config["image_gen_dir"] + "*.png"
    image_files = sorted(glob.glob(file_path))
    first_image = Image.open(image_files[0])
    frames = [Image.open(image) for image in image_files[1:]]
    first_image.save(os.path.join(config["image_dir"], config["gif_name"]),
                     save_all=True, append_images=frames, duration=100, loop=0)


if __name__ == '__main__':
    modelConfig = {
        "epoch": 70,
        "batch_size": 128,
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
        "img_size": 32,
        "grad_clip": 1.,
        "w": 1.8,
        "nrow": 10,
        "num_class": 10,
        "model_dir": "../00_assets/model_cifar10/",
        "image_dir": "../00_assets/img_cifar10/",
        "image_gen_dir": "../00_assets/img_cifar10/images/",
        "training_weight": None,
        "eval_weight": "ckpt_63.pth",
        "noisy_name": "img_noisy.png",
        "generate_name": "img_generate.png",
        "gif_name": "img_generate.gif",
        "raw_name": "img_raw.png"
    }

    os.makedirs(modelConfig["model_dir"], exist_ok=True)
    os.makedirs(modelConfig["image_dir"], exist_ok=True)
    os.makedirs(modelConfig["image_gen_dir"], exist_ok=True)

    train(modelConfig)
    generate(modelConfig)
