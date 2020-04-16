class Client:
    def __init__(self, client_id, dataloader, device='cpu'):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
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
        self.model.to(self.device)
        optimizer = optimizer(self.model.parameters(), **optimizer_args)
        for i in range(local_epoch):
            for img, target in self.dataloader:
                img = img.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                logits = self.model(img)
                loss = loss_fn(logits, target)

                loss.backward()
                optimizer.step()
        self.model.to("cpu")