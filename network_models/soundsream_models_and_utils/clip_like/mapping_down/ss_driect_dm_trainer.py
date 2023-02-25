import gc
import os
from pathlib import Path
from typing import Sized, Optional, Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F

from network_models.soundsream_models_and_utils.ss___util_class_batches_sampler import ClassBatchesSampler
from network_models.soundsream_models_and_utils.ss_encoded_dataset import ss_encoded_dataset_full
from network_models.w2v_emotion_model.custom_collator import DataCollatorCTCWithPadding
from utils.eval_utils import classificationReport, confusion_matrix




class SSDirectDMTrainer():
    def __init__(
            self,
            model: nn.Module,
            dataset: ss_encoded_dataset_full,
            device: str,
            num_epochs: int,
            batch_size: int,
            lr=2e-4,
            save_model_every=1000,
            loss_fn=nn.CrossEntropyLoss(),
            model_path="content/customModel/soundstream/",
            is_encoder= False

    ):
        self.is_encoder = is_encoder
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
        labels = self.dataset.encoded_dataset.encodedData[self.dataset.encoded_dataset.labelcolumn].to_numpy()
        dataloader = DataLoader(self.dataset,  num_workers=2, batch_sampler=ClassBatchesSampler(labels, num_class_samples=2))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-7)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        Path(self.model_path).mkdir(parents=True, exist_ok=True)

        for t in range(self.num_epochs):
            if (t % self.save_model_every == 0):
                torch.save(self.model.state_dict(), self.model_path + f"encoder_{t}.pth")
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train_loop(dataloader, self.model, self.loss_fn, optimizer)
            gc.collect()

    def train_loop(self, dataloader, model, loss_fn, optimizer):
        fulllos = 0
        size = len(dataloader.dataset)
        for batch, (X, z) in enumerate(dataloader):
            z = z.to(self.device)
            X = X.to(self.device)
            if (len(X) % 2 != 0):
                continue

            if self.is_encoder: # X.shape = (14,512,175)
                X = torch.squeeze(X, dim=1)
                X = torch.transpose(X, 1, 2)
                #X = X/X.norm(dim=1, keepdim=True)

            #z1, z2 = torch.tensor_split(z, 2)
            pred = model(X)
            pred1, pred2 = torch.tensor_split(pred, 2)

            pred1 = pred1 / pred1.norm(dim=-1, keepdim=True)

            pred2 = pred2 / pred2.norm(dim=-1, keepdim=True)

            # cos_sim = torch.cosine_similarity(pred1, pred2)
            # gt = torch.ones(7, device=self.device)
            # Todo: norm like clip
            dot_products = torch.matmul(pred1, pred2.T)
            #cos_sim_labels = torch.tensor(np.identity(7)).to(self.device)
            gt = torch.arange(len(pred1), dtype=torch.long).to(self.device)

            b1_to_b2_sim = dot_products
            b2_to_b1_sim = dot_products.t()


            loss = (loss_fn(b1_to_b2_sim, gt) + loss_fn(b2_to_b1_sim, gt))*0.5

            fulllos += loss.item()
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                # print(dp)

            # if ((size //( 2*7))-2 == batch):
            #     print(b1_to_b2_sim)
            #     print(b2_to_b1_sim)
                #print(cos_sim_labels)
                # print(target)
                # print(target_sm)

        print(fulllos)
