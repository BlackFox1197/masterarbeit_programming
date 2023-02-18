import gc
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from network_models.w2v_emotion_model.custom_collator import DataCollatorCTCWithPadding
from utils.eval_utils import classificationReport, confusion_matrix


class SSEncoderTrainer():
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
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        Path(self.model_path).mkdir(parents=True, exist_ok=True)

        for t in range(self.num_epochs):
            if(t % self.save_model_every == 0):
                torch.save(self.model.state_dict(), self.model_path + f"encoder_{t}.pth")
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train_loop(dataloader, self.model, self.loss_fn, optimizer)
            gc.collect()




    def train_loop(self, dataloader, model, loss_fn, optimizer):


        size = len(dataloader.dataset)
        for batch, (X, z) in enumerate(dataloader):

            target_output = torch.tensor(np.identity(len(X))).to(self.device)
            X = X.to(self.device)
            X = torch.squeeze(X, dim=1)
            X = torch.transpose(X, 1,2)
            # Compute prediction and loss
            #z = z.to(self.device)
            pred = model(X)

            # generate dotproducts
            dp = torch.matmul(pred, pred.T)



            real_pred = F.softmax(
                dp/
                torch.sqrt(
                    torch.abs(
                        torch.sum(dp, dim=0))), dim=0).to(self.device)
            loss = loss_fn(real_pred, target_output)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                print(real_pred)
