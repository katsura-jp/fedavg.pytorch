import os
import logging
import pickle

log = logging.getLogger(__name__)

import torch
from torch.utils.tensorboard import SummaryWriter
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
    seed_everything(cfg.seed)
    log.info("\n" + cfg.pretty())

    model = eval(cfg.model.classname)(**cfg.model.args)
    writer = SummaryWriter(log_dir=os.path.join(cfg.savedir, "tf"))
    federater = eval(cfg.fed.classname)(model=model,
                                        optimizer=eval(cfg.optim.classname),
                                        optimizer_args=cfg.optim.args,
                                        num_clients=cfg.K,
                                        batchsize=cfg.B,
                                        fraction=cfg.C,
                                        local_epoch=cfg.E,
                                        iid=cfg.iid,
                                        device=cfg.device,
                                        writer=writer)

    federater.fit(cfg.n_round)

    with open(os.path.join(cfg.savedir, "result.pkl"), "wb") as f:
        pickle.dump(federater.result, f)


if __name__ == "__main__":
    main()
