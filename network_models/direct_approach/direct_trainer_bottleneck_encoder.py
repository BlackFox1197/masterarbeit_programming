import gc
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class DirectAutoencoderTrainer():
    def __init__(
            self,
            model: nn.Module,
            dataset: Dataset,
            device: str,
            num_epochs: int,
            batch_size: int,
            lr=2e-4,
            save_model_every=1000,
            loss_fn=nn.CrossEntropyLoss(),
            model_path = "content/customModel/soundstream/"

    ):
        self.model_path = model_path
        self.model = model
        self.dataset = dataset
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.save_model_every = save_model_every
        self.loss_fn = loss_fn





    def train(self):
        dataloader = DataLoader(self.dataset, shuffle=True, batch_size=self.batch_size, num_workers=2)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        #optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        Path(self.model_path).mkdir(parents=True, exist_ok=True)

        for t in range(self.num_epochs):
            if(t % self.save_model_every == 0):
                torch.save(self.model.state_dict(), self.model_path + f"encoder_{t}.pth")
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train_loop(dataloader, self.model, self.loss_fn, optimizer)
            gc.collect()




    def train_loop(self, dataloader, model, loss_fn, optimizer):

        main_loss = 0
        size = len(dataloader.dataset)
        for batch, (X, z) in enumerate(dataloader):
            X = torch.squeeze(X, dim=1)
            X = X.to(self.device)
            #X = X/X.norm(dim=2, keepdim=True)
            pred = model(X)

            #pred = pred/pred.norm(dim=1, keepdim=True)
            #normedX = X.T/X.T.norm(dim=1, keepdim=True)
            normedX = X.T
            target_output = torch.tensor(np.identity(len(X))).to(self.device)
            target_output = torch.arange(0, len(X)).to(self.device)
            dp = torch.softmax(torch.matmul(pred, normedX), dim=1)

            loss = loss_fn(dp, target_output)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            main_loss += loss.item()

            if batch*self.batch_size % 1000 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        print(main_loss)
