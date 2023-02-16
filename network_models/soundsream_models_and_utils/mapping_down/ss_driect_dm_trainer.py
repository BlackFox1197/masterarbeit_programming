import gc
import os
from pathlib import Path
from typing import Sized, Optional, Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F

from network_models.w2v_emotion_model.custom_collator import DataCollatorCTCWithPadding
from utils.eval_utils import classificationReport, confusion_matrix


class CustomBatchSampler(object):
    def __init__(self, targets, num_samples=1):
        self.targets = targets
        self.num_samples = num_samples
        self.num_classes = len(np.unique(self.targets))

    def __iter__(self):
        for i in range(self.num_classes):
            idxs = np.where(self.targets == i)[0]
            np.random.shuffle(idxs)
            num_batches = len(idxs) // self.num_samples
            for j in range(num_batches):
                yield idxs[j*self.num_samples:(j+1)*self.num_samples]

    def __len__(self):
        return len(self.targets) // self.num_samples


class SSDirectDMTrainer():
    def __init__(
            self,
            model: nn.Module,
            dataset: Dataset,
            device: str,
            num_epochs: int,
            batch_size: int,
            lr=2e-4,
            save_model_every=1000,
            loss_fn=nn.CosineEmbeddingLoss(),
            model_path="content/customModel/soundstream/"

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
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=2, batch_sampler=BatchSamplerForDiffClasses)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        Path(self.model_path).mkdir(parents=True, exist_ok=True)

        for t in range(self.num_epochs):
            if (t % self.save_model_every == 0):
                torch.save(self.model.state_dict(), self.model_path + f"encoder_{t}.pth")
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train_loop(dataloader, self.model, self.loss_fn, optimizer)
            gc.collect()

    def train_loop(self, dataloader, model, loss_fn, optimizer):

        size = len(dataloader.dataset)
        for batch, (X, z) in enumerate(dataloader):

            X = X.to(self.device)
            if (len(X) % 2 != 0):
                continue
            # X = torch.squeeze(X, dim=1)
            # X = torch.transpose(X, 1,2)
            # Compute prediction and loss
            # z = z.to(self.device)
            pred = model(X)

            z = np.argmax(z, axis=1)

            # generate dotproducts
            # dp = torch.matmul(pred, pred.T)
            # dp = dp  / (pred.norm(dim=-1, keepdim=True) * pred.norm(dim=-1, keepdim=True))

            pred1, pred2 = pred.split(split_size=len(pred) // 2)
            z1, z2 = np.array_split(z, 2)

            sim = torch.cosine_similarity(pred1, pred2)
            target = torch.tensor((np.equal(z1, z2) * 2) - 1).to(self.device)
            loss = loss_fn(pred1, pred2, target=target)

            # dp = (dp +1) /2
            # target = torch.Tensor((np.equal.outer(z, z)*2) - 1).to(self.device)
            target_iden = torch.tensor(np.identity(len(X))).to(self.device)

            # dp = dp /torch.sum(dp, dim=1)
            # target_sm = target /torch.sum(target, dim=1)
            # target_iden = target_iden

            # loss = 0.8 * loss_fn(dp, target) #+ 0.2 * loss_fn(dp, target_iden)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                # print(dp)

            if (size // self.batch_size - 1 == batch):
                print(sim)
                print(target)
                # print(target_sm)
