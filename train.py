import os
import logging

import torch
import hydra
from omegaconf import DictConfig

from src.models import *
from src.fed_zoo import *
from src.utils import *
from src.utils.data import get_mnist_data

# @hydra.main(config_path="./config/config.yaml", strict=True)
# def main(cfg: DictConfig):
def main():
    model = MLP(784, 10, 200)

    federater = FedAvg(model,
                       optimizer=torch.optim.SGD,
                       optimizer_args={'lr': 0.01},
                       num_clients=100,
                       batchsize=50,
                       fraction=0.1,
                       local_epoch=4)

    federater.fit(10)
    pass


if __name__ == "__main__":
    main()