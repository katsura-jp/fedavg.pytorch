import os
import logging

log = logging.getLogger(__name__)

import torch
import hydra
from omegaconf import DictConfig
from torch.optim import *

from src.models import *
from src.fed_zoo import *
from src.utils import *
from src.utils.data import get_mnist_data

@hydra.main(config_path="./config/config.yaml", strict=True)
def main(cfg: DictConfig):
    os.chdir(cfg.root)
    log.info("\n" + cfg.pretty())

    model = eval(cfg.model.classname)(**cfg.model.args)

    federater = eval(cfg.fed.classname)(
                        model=model,
                        optimizer=eval(cfg.optim.classname),
                        optimizer_args=cfg.optim.args,
                        num_clients=cfg.K,
                        batchsize=cfg.B,
                        fraction=cfg.C,
                        local_epoch=cfg.E)

    federater.fit(cfg.n_round)
    pass


if __name__ == "__main__":
    main()