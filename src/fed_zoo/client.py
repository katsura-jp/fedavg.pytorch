class Client:
    def __init__(self, client_id, dataloader):
        self.client_id = client_id
        self.dataloader = dataloader
        self.__model = None

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    def client_update(self, optimizer, optimizer_args, local_epoch, loss_fn):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataloader.dataset)


class FedAvgClient(Client):
    def client_update(self, optimizer, optimizer_args, local_epoch, loss_fn):
        self.model.train()
        optimizer = optimizer(self.model.parameters(), **optimizer_args)
        for i in range(local_epoch):
            for img, target in self.dataloader:
                optimizer.zero_grad()
                logits = self.model(img)
                loss = loss_fn(logits, target)

                loss.backward()
                optimizer.step()