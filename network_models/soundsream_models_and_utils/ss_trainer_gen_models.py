import gc
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from network_models.clip.trainer_and_utils.ss___util_class_batches_sampler import ClassBatchesSampler
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
            model_path = "content/customModel/soundstream/",
            regularize_dims = False

    ):
        self.regularize_dims = regularize_dims
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

    num_class_samples = 2



    def train(self):
        if(self.regularize_dims):
            labels =(self.train_dataset.dataset.encoded_dataset.encodedData[self.train_dataset.dataset.encoded_dataset.labelcolumn].to_numpy())[self.train_dataset.indices]
            labels_test = (self.test_dataset.dataset.encoded_dataset.encodedData[self.test_dataset.dataset.encoded_dataset.labelcolumn].to_numpy())[self.test_dataset.indices]
            train_dataloader = DataLoader(self.train_dataset,  num_workers=2, batch_sampler=ClassBatchesSampler(labels, num_class_samples=2))
            test_dataloader = DataLoader(self.test_dataset,  num_workers=2, batch_sampler=ClassBatchesSampler(labels_test, num_class_samples=2))
        else:
            train_dataloader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=2)
            test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=1e-4)
        #optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        Path(self.model_path).mkdir(parents=True, exist_ok=True)
        highest_acc = 0
        higest_epoch = None
        higest_true = []
        higest_pred = []


        epoch_train_losses = []
        epoch_eval_losses = []
        epoch_eval_accs = []


        for t in range(self.num_epochs):
            if(t % self.save_model_every == 0):
                torch.save(self.model.state_dict(), self.model_path + f"emo_reco_{t}.pth")
            print(f"Epoch {t + 1}\n-------------------------------")
            # this is the trainloop
            epoch_train_losses = epoch_train_losses +  [self.train_loop(train_dataloader, self.model, self.loss_fn, optimizer, t)]

            # --------------------- testloop and evaluation- ---------------
            acc, true, preds, loss = self.test_loop(test_dataloader, self.model, self.loss_fn)
            epoch_eval_losses = epoch_eval_losses + [loss]
            epoch_eval_accs = epoch_eval_accs + [acc]

            if(acc > highest_acc):
                old_acc = highest_acc
                old_epoch = higest_epoch
                highest_acc, higest_epoch, higest_true, higest_pred = acc, t, true, preds
                # this is for saving the best accuracy up until now
                if(self.save_highest_acc_min_acc != None and self.save_highest_acc_min_acc < acc):
                    #old_acc = highest_acc if highest_acc != 0 else None
                    self.save_best(self.model, acc, t, true, preds, old_acc, old_epoch)
                    highest_acc, higest_epoch = acc, t
            gc.collect()

            information_dict = {"eval_losses" : epoch_eval_losses, "train_losses": epoch_train_losses, "eval_accs": epoch_eval_accs}
            with open(f'{self.model_path}_statistics.json', 'w', encoding='utf-8') as f:
                json.dump(information_dict, f, ensure_ascii=False, indent=4)

        return highest_acc, higest_epoch, higest_true, higest_pred




    def train_loop(self, dataloader, model, loss_fn, optimizer, epoch):
        fullLoss = 0
        batch = 0
        size = len(dataloader.dataset)
        for batch, (X, z) in enumerate(dataloader):

            if (len(X) % 2 != 0):
                continue
            X = X.to(self.device)
            # Compute prediction and loss
            z = z.to(self.device)

            if(self.regularize_dims):
                pred_dims, pred = model(X, return_with_dims = True)
                cosine_loss = self.calculate_cosine_loss(pred_dims, loss_fn)
                classification_loss = loss_fn(pred, torch.argmax(z, dim=1))
                #loss = classification_loss*(max(0.7-(epoch*self.lr/2), 0.4)) + cosine_loss*(min(0.3+(epoch*self.lr/2),0.6))
                loss = classification_loss*0.8 + cosine_loss*0.2

            else:
                pred = model(X)
                loss = loss_fn(pred, z)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            fullLoss += loss.item()
            batch += 1

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                if (self.regularize_dims):
                    print(f"[{current:>5d}/{size:>5d}] total-loss: {loss:>7f}; classification loss: {classification_loss:>8f};  cosine similarity loss: {cosine_loss:>8f}")
                else:
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            #gc.collect()
        return fullLoss/batch


    def calculate_cosine_loss(self, pred_dims, loss_fn):
        pred1, pred2 = torch.tensor_split(pred_dims, 2)
        pred1, pred2 = pred1/pred1.norm(dim=1, keepdim=True), pred2/pred2.norm(dim=1, keepdim=True)


        dot_products = torch.matmul(pred1, pred2.T)  # / (pred1.norm(dim=-1, keepdim=True) * pred2.norm(dim=-1, keepdim=True))
        gt = torch.arange(len(pred1), dtype=torch.long).to(self.device)

        b1_to_b2_sim = torch.softmax(dot_products, dim=1)
        b2_to_b1_sim = torch.softmax(dot_products.t(), dim=1)

        return (loss_fn(b1_to_b2_sim, gt) + loss_fn(b2_to_b1_sim, gt))*0.5

    def test_loop(self, dataloader, model, loss_fn): #TODO: Add more than accuracy (recall, precision)
        size = len(dataloader.dataset)
        size_reg = len(dataloader)
        test_loss, correct, cosine_loss_full, classification_loss_full = 0, 0,0,0
        true, preds = [], []
        batches = 0

        with torch.no_grad():
            for X, labels in dataloader:
                X = X.to(self.device)
                labels = labels.to(self.device)

                if (self.regularize_dims):
                    pred_dims, pred = model(X, return_with_dims=True, eval_mode = True)
                    cosine_loss = self.calculate_cosine_loss(pred_dims, loss_fn)
                    classification_loss = loss_fn(pred, labels)
                    loss = classification_loss * 0.7 + cosine_loss * 0.3

                else:
                    pred = model(X, eval_mode = True)
                    loss = loss_fn(pred, labels)

                true = true + [torch.squeeze(a.nonzero()).item() for a in labels]
                preds = preds + pred.argmax(1).cpu().numpy().tolist()
                test_loss += loss.item()
                if(self.regularize_dims):
                    cosine_loss_full += cosine_loss.item()
                    classification_loss_full += classification_loss.item()
                labels = torch.tensor([a.nonzero() for a in labels]).to(self.device)
                correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
                batches += 1

        if(self.labelList is not None):
            classificationReport(true, preds, self.labelList)


        test_loss /= batches
        cosine_loss_full /=batches
        classification_loss_full /=batches
        if(self.regularize_dims):
            correct /= (size_reg//len(self.labelList*self.num_class_samples))*(len(self.labelList*self.num_class_samples))
        else:
            correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        if(self.regularize_dims):
            print(f"classification loss: {classification_loss_full:>8f};  cosine similarity loss: {cosine_loss_full:>8f} \n")
        return correct, true, preds, test_loss



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
            if os.path.exists(old_path+".pth"):
                os.remove(old_path+".pth")
                print(f"Old best model deleted from \"{old_path}\"")
                old_string = f"Old accuracy: {(100 * old_acc):>0.1f},"
            if os.path.exists(old_path+".md"):
                os.remove(old_path+".md")


        torch.save(model.state_dict(), new_path+".pth")
        print(f"New best model saved to \"{new_path}\"! {old_string} new accuracy: {(100 * acc):>0.1f}")


    @staticmethod
    def genAndSaveEvaluation(filename, ground_truth, pred, acc, epoch, modelName, labelList):
        print("Generating Report... \n")
        save_str = "#" + modelName + "\n"
        save_str += "## Evaluations: \n"
        save_str += "```"+classificationReport(true_codes=ground_truth, pred_codes=pred,sortedLabelStrings=labelList, printReport=False, return_string=True) +"```\n \n"
        save_str += "```\n"+confusion_matrix(true_codes=ground_truth, pred_codes=pred,sortedLabelStrings=labelList, printReport=False) + "```\n"
        save_str += f"Max Accuracy: {acc} in epoch {epoch}"

        file = open(filename+".md", "w")
        file.write(save_str)
        file.close()
        print(f"Report saved to: {filename}.md \n")