import gc

import torch

from network_models.soundsream_models_and_utils.ss_encoded_dataset import ss_encoded_dataset_full
from network_models.soundsream_models_and_utils.ss_model_conv import SSConvModel3Sec
from network_models.soundsream_models_and_utils.ss_model_dim_red import SSDimRedModel
from network_models.soundsream_models_and_utils.ss_model_flat import SSFlatModel
from network_models.soundsream_models_and_utils.ss_trainer_gen_models import SSGenModelTrainer
from utils.audio_dataset_utils import train_val_dataset


class ExperimentsTrainer:


    def __init__(self, dataset: ss_encoded_dataset_full, models_dir = "content/soundstream/experiments/",
                 trials_per_model_type: int = 1, epochs_per_model = 1000, start_lr = 1e-4, lr_quotient = 2,
                 batch_size =8, device = "cuda" if torch.cuda.is_available() else "cpu", save_model_every = 50):
        self.dataset =dataset
        self.models_dir =models_dir
        self.trials_per_model_type =trials_per_model_type
        self.epochs_per_model =epochs_per_model
        self.start_lr =start_lr
        self.lr_quotient =lr_quotient
        self.batch_size =batch_size
        self.device =device
        self.label_list = self.dataset.encoded_dataset.label_list
        self.safe_model_every = save_model_every
    def train_em(self): # TODO: save model at the end
        for trail in range(self.trials_per_model_type):
            lr = self.start_lr /  (self.lr_quotient ** trail)
            self.run_conv_model_test(lr, self.epochs_per_model, trail)
            gc.collect()
            self.run_dim_red_model_test(lr, self.epochs_per_model, trail)
            gc.collect()
            #self.run_flat_model_test(lr, self.epochs_per_model, trail)
            gc.collect()


    def run_conv_model_test(self, lr, epochs, current_run):
        x_size, y_size = self.dataset[0][0][0].shape
        model = SSConvModel3Sec(num_emotions=len(self.label_list), xSize=x_size, ySize=y_size).to(self.device)
        save_dir = self.models_dir + f"Run_Nr_{current_run}_conv/"
        return self.run_model_test(lr, epochs, model, save_dir)

    def run_dim_red_model_test(self, lr, epochs, current_run):
        bs = self.batch_size * 2
        model = SSDimRedModel(num_emotions=len(self.label_list)).to(self.device)
        save_dir = self.models_dir + f"Run_Nr_{current_run}_dimred/"
        return self.run_model_test(lr, epochs, model, save_dir, bs=bs)

    def run_flat_model_test(self, lr, epochs, current_run):
        #bs = max(self.batch_size // 2, 1)
        bs = self.batch_size
        model = SSFlatModel(num_emotions=len(self.label_list)).to(self.device)
        save_dir = self.models_dir + f"Run_Nr_{current_run}_flat/"
        return self.run_model_test(lr, epochs, model, save_dir, bs=bs)


    def run_model_test(self, lr, epochs, model, save_dir, bs = None):
        trainDS, testDs = train_val_dataset(self.dataset, val_split=0.1, seed=100)
        trainer = SSGenModelTrainer(lr=lr, num_epochs=epochs, model=model, train_dataset=trainDS,
                                    eval_dataset=testDs,
                                    device=self.device, labelList=self.label_list,
                                    batch_size=self.batch_size if bs is None else bs, save_model_every=self.safe_model_every, save_highest_acc_min_acc=0.6,
                                    model_path = save_dir)
        return trainer.train()