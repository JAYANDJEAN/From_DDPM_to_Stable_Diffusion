import os
from typing import Dict
import numpy as np

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from utils import GaussianDiffusionSampler, GaussianDiffusionTrainer, GradualWarmupScheduler
from unet import UNet


def train(config: Dict):
    device = torch.device(config["device"])
    # dataset
    dataset = CIFAR10(root='../00_assets/datasets', train=True, download=True,
                      transform=Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)]))
    dataloader = DataLoader(dataset, batch_size=config["batch_size"],
                            shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(channel_img=3, channel_base=config["channel"], channel_mults=config["channel_mult"],
                     num_res_blocks=config["num_res_blocks"], dropout=config["dropout"]).to(device)
    if config["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            config["save_dir"], config["training_load_weight"]), map_location=device), strict=False)
        print("Model weight load down.")
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=config["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=config["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=config["multiplier"],
                                             warm_epoch=config["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(net_model, config["beta_1"], config["beta_T"], config["T"]).to(device)

    # start training
    for e in range(config["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device) + 1
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)
                loss = trainer(x_0, labels).sum() / b ** 2.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), config["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        os.makedirs(config["save_dir"], exist_ok=True)
        torch.save(net_model.state_dict(), os.path.join(config["save_dir"], 'ckpt_' + str(e) + ".pth"))


def generate(config: Dict):
    device = torch.device(config["device"])
    # load model and evaluate
    with torch.no_grad():
        step = int(config["batch_size"] // 10)
        labelList = []
        k = 0
        for i in range(1, config["batch_size"] + 1):
            labelList.append(torch.ones(size=[1]).long() * k)
            if i % step == 0:
                if k < 10 - 1:
                    k += 1
        labels = torch.cat(labelList, dim=0).long().to(device) + 1
        print("labels: ", labels)
        model = UNet(channel_img=3, channel_base=config["channel"], channel_mults=config["channel_mult"],
                     num_res_blocks=config["num_res_blocks"], dropout=config["dropout"]).to(device)
        ckpt = torch.load(os.path.join(config["save_dir"], config["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, config["beta_1"], config["beta_T"], config["T"], w=config["w"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[config["batch_size"], 3, config["img_size"], config["img_size"]], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        os.makedirs(config["sampled_dir"], exist_ok=True)
        save_image(saveNoisy, os.path.join(config["sampled_dir"], config["sampledNoisyImgName"]), nrow=config["nrow"])
        sampledImgs = sampler(noisyImage, labels)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        print(sampledImgs)
        save_image(sampledImgs, os.path.join(config["sampled_dir"], config["sampledImgName"]), nrow=config["nrow"])


if __name__ == '__main__':
    modelConfig = {
        "state": "train",  # or eval
        "epoch": 70,
        "batch_size": 80,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "w": 1.8,
        "save_dir": "../00_assets/model_cifar10/",
        "training_load_weight": None,
        "test_load_weight": "ckpt_63.pth",
        "sampled_dir": "../00_assets/img_cifar10/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 8
    }
    train(modelConfig)
    generate(modelConfig)
