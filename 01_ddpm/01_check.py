import torch
from unet import UNet
import torch.nn as nn

batch_size = 128
n_step = 1000
n_class = 10
n_time_embed = 100


def check_unet_output():
    x0 = torch.rand(batch_size, 1, 28, 28)
    unet = UNet(channels=[1, 10, 10, 20, 20, 40, 40, 64],
                p_dropouts=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                time_embed_size=n_time_embed,
                num_classes=n_class,
                use_down=True,
                use_attention=True)
    x_recon = unet(x0, embed_t, embed_c)
    assert x_recon.shape == x0.shape

    x1 = torch.rand(batch_size, 3, 32, 35)
    unet = UNet(channels=[3, 32, 64, 128],
                p_dropouts=[0.0, 0.0, 0.0],
                time_embed_size=100,
                num_classes=n_class,
                use_down=True,
                use_attention=True)
    x_recon = unet(x1, embed_t, embed_c)
    assert x_recon.shape == x1.shape


if __name__ == '__main__':
    t0 = torch.randint(0, n_step, (batch_size,))
    y0 = torch.randint(0, n_class, (batch_size,))
    embed_c = nn.functional.one_hot(y0.squeeze(), n_class).float()
    embed_t = torch.rand(batch_size, n_time_embed)

    check_unet_output()



