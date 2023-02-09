import gc
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from network_models.w2v_emotion_model.custom_collator import DataCollatorCTCWithPadding
from utils.eval_utils import classificationReport, confusion_matrix


class SSGenModelTrainer():
    def __init__(
            self,
            model: nn.Module,
            train_dataset: Dataset,
            eval_dataset: Dataset,
            device: str,
            num_epochs: int,
            batch_size: int,
            labelList = None,
            lr=2e-4,
            save_model_every=1000,
            save_highest_acc_min_acc: float | None = 0.7,
            loss_fn=nn.CrossEntropyLoss(),
            model_path = "content/customModel/soundstream/"

    ):
        self.model_path = model_path
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = eval_dataset
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # this parameter is ment to activate saving the model with the highest accuracy, starting from the accuracy given by this parameter
        self.save_highest_acc_min_acc = save_highest_acc_min_acc
        self.lr = lr
        self.save_model_every = save_model_every
        self.loss_fn = loss_fn
        self.labelList = labelList




    def train(self):
        train_dataloader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=2)
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        #optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        Path(self.model_path).mkdir(parents=True, exist_ok=True)
        highest_acc = 0
        higest_epoch = None
        higest_true = []
        higest_pred = []

        for t in range(self.num_epochs):
            if(t % self.save_model_every == 0):
                torch.save(self.model.state_dict(), self.model_path + f"emo_reco_{t}.pth")
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train_loop(train_dataloader, self.model, self.loss_fn, optimizer)
            acc, true, preds = self.test_loop(test_dataloader, self.model, self.loss_fn)

            if(acc > highest_acc):
                highest_acc, higest_epoch, higest_true, higest_pred = acc, t, true, preds
                # this is for saving the best accuracy up until now
                if(self.save_highest_acc_min_acc != None and self.save_highest_acc_min_acc < acc):
                    #old_acc = highest_acc if highest_acc != 0 else None
                    old_acc = highest_acc
                    self.save_best(self.model, acc, t, true, preds, old_acc, higest_epoch)
                    highest_acc, higest_epoch = acc, t
            gc.collect()
        return highest_acc, higest_epoch, higest_true, higest_pred




    def train_loop(self, dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, z) in enumerate(dataloader):

            X = X.to(self.device)
            # Compute prediction and loss
            z = z.to(self.device)
            pred = model(X)

            loss = loss_fn(pred, z)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            #gc.collect()


    def test_loop(self, dataloader, model, loss_fn): #TODO: Add more than accuracy (recall, precision)
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        true, preds = [], []

        with torch.no_grad():
            for X, labels in dataloader:
                X = X.to(self.device)
                labels = labels.to(self.device)
                pred = model(X)

                true = true + [torch.squeeze(a.nonzero()).item() for a in labels]
                preds = preds + pred.argmax(1).cpu().numpy().tolist()
                test_loss += loss_fn(pred, labels.to(self.device)).item()
                labels = torch.tensor([a.nonzero() for a in labels]).to(self.device)
                correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

        if(self.labelList is not None):
            classificationReport(true, preds, self.labelList)


        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return correct, true, preds



    def save_best(self, model, acc, epoch, ground_truth, pred, old_acc = None, old_epoch = None, ):
        """saves model with accuracy and deletes the old best model"""

        def gen_filename(acc_in, epoch_in):
            return self.model_path + f"emo_reco_best_ep{epoch_in}_acc_{acc_in*100:.0f}"

        modelName = f"{model.__class__}".split('.')[-1].split("'>")[0]
        self.genAndSaveEvaluation(gen_filename(acc, epoch), ground_truth, pred, acc, epoch, modelName, self.labelList)

        new_path = gen_filename(acc, epoch)
        old_string = ""
        if old_acc is not None:
            old_path = gen_filename(old_acc, old_epoch)
            if os.path.exists(old_path):
                os.remove(old_path)
                print(f"Old best model deleted from \"{old_path}\"")
                old_string = f"Old accuracy: {(100 * old_acc):>0.1f},"
            if os.path.exists(old_path+".md"):
                os.remove(old_path+".md")


        torch.save(model.state_dict(), new_path)
        print(f"New best model saved to \"{new_path}\"! {old_string} new accuracy: {(100 * acc):>0.1f}")


    @staticmethod
    def genAndSaveEvaluation(filename, ground_truth, pred, acc, epoch, modelName, labelList):
        print("Generating Report... \n")
        save_str = "#" + modelName + "\n"
        save_str += "## Evaluations:"
        save_str += "```"+classificationReport(true_codes=ground_truth, pred_codes=pred,sortedLabelStrings=labelList, printReport=False, return_string=True) +"```\n \n"
        save_str += "```"+confusion_matrix(true_codes=ground_truth, pred_codes=pred,sortedLabelStrings=labelList, printReport=False) + "```\n"
        save_str += f"Max Accuracy: {acc} in epoch {epoch}"

        file = open(filename+".md", "w")
        file.write(save_str)
        file.close()
        print(f"Report saved to: {filename}.md \n")