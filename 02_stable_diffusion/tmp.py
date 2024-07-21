import yaml
from torch import nn
from torch.optim import Adam


with open('../00_assets/cifar.yaml', 'r') as file:
    config = yaml.safe_load(file)
print(type(config['channels']))
