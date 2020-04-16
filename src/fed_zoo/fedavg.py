import logging
log = logging.getLogger(__name__)

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from src.fed_zoo.base import FedBase
from src.fed_zoo.client import FedAvgClient as Client
from src.fed_zoo.center_server import FedAvgCenterServer as CenterServer


class FedAvg(FedBase):
    def __init__(self,
                 model,
                 optimizer,
                 optimizer_args,
                 num_clients=200,
                 batchsize=50,
                 fraction=1,
                 local_epoch=1,
                 iid=False,
                 writer=None):
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args

        self.num_clients = num_clients  # K
        self.batchsize = batchsize  # B
        self.fraction = fraction  # C, 0 < C <= 1
        self.local_epoch = local_epoch  # E

        local_datasets, test_dataset = self.create_mnist_datasets(
            num_clients, shard_size=300, iid=iid)
        local_dataloaders = [
            DataLoader(dataset,
                       num_workers=0,
                       batch_size=batchsize,
                       shuffle=True) for dataset in local_datasets
        ]

        self.clients = [
            Client(k, local_dataloaders[k]) for k in range(num_clients)
        ]
        self.total_data_size = sum([len(client) for client in self.clients])
        self.aggregation_weights = [
            len(client) / self.total_data_size for client in self.clients
        ]

        test_dataloader = DataLoader(test_dataset,
                                     num_workers=0,
                                     batch_size=batchsize)
        self.center_server = CenterServer(model, test_dataloader)

        self.loss_fn = CrossEntropyLoss()

        self.writer = writer

        self._round = 0

    def fit(self, num_round):
        self._round = 0
        self.validation_step()
        for t in range(num_round):
            self._round = t + 1
            self.train_step()
            self.validation_step()

    def train_step(self):
        self.send_model()
        n_sample = max(int(self.fraction * self.num_clients), 1)
        sample_set = np.random.randint(0, self.num_clients, n_sample)
        for k in iter(sample_set):
            self.clients[k].client_update(self.optimizer, self.optimizer_args,
                                          self.local_epoch, self.loss_fn)
        self.center_server.aggregation(self.clients, self.aggregation_weights)

    def send_model(self):
        for client in self.clients:
            client.model = self.center_server.send_model()

    def validation_step(self):
        test_loss, accuracy = self.center_server.validation(self.loss_fn)
        log.info(
            f"[Round: {self._round: 04}] Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )
        if self.writer is not None:
            self.writer.add_scalar("val/loss", test_loss, self._round)
            self.writer.add_scalar("val/accuracy", accuracy, self._round)
