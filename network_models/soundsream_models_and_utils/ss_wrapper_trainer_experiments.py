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
                 batch_size =8, device = "cuda" if torch.cuda.is_available() else "cpu",
                 save_model_every = 50, save_highest_acc_min_acc=0.6, seed = 200, regularize_dims = False):

        self.regularize_dims = regularize_dims
        self.seed = seed
        self.dataset =dataset
        self.save_highest_acc_min_acc =save_highest_acc_min_acc
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
            #highest_acc_c, higest_epoch_c, higest_true_c, higest_pred_c = self.run_conv_model_test(lr, self.epochs_per_model, trail)
            ## generate Report
            #SSGenModelTrainer.genAndSaveEvaluation(f"{self.models_dir}/Run_Nr_{trail}/conv", higest_true_c, higest_pred_c, highest_acc_c, higest_epoch_c, "Convolutional", self.dataset.encoded_dataset.label_list)

            gc.collect()
            highest_acc_dr, higest_epoch_dr, higest_true_dr, higest_pred_dr = self.run_dim_red_model_test(lr, self.epochs_per_model, trail)
            # generate Report
            SSGenModelTrainer.genAndSaveEvaluation(f"{self.models_dir}/Run_Nr_{trail}/dimred", higest_true_dr, higest_pred_dr, highest_acc_dr, higest_epoch_dr, "Dimension Reduced", self.dataset.encoded_dataset.label_list)
            gc.collect()
            #highest_acc_f, higest_epoch_f, higest_true_f, higest_pred_f = self.run_flat_model_test(lr, self.epochs_per_model, trail)
            ## generate Report
            #SSGenModelTrainer.genAndSaveEvaluation(f"Run_Nr_{trail}/flat", higest_true_f, higest_pred_f, highest_acc_f, higest_epoch_f, "Flatted Model", self.dataset.encoded_dataset.label_list)
            gc.collect()


    def run_conv_model_test(self, lr, epochs, current_run):
        x_size, y_size = self.dataset[0][0][0].shape
        torch.manual_seed(self.seed)
        model = SSConvModel3Sec(num_emotions=len(self.label_list), xSize=x_size, ySize=y_size).to(self.device)
        save_dir = self.models_dir + f"/Run_Nr_{current_run}/conv/"
        return self.run_model_test(lr, epochs, model, save_dir)

    def run_dim_red_model_test(self, lr, epochs, current_run):
        bs = self.batch_size
        torch.manual_seed(self.seed)
        model = SSDimRedModel(num_emotions=len(self.label_list)).to(self.device)
        save_dir = self.models_dir + f"/Run_Nr_{current_run}/dimred/"
        return self.run_model_test(lr, epochs, model, save_dir, bs=bs)

    def run_flat_model_test(self, lr, epochs, current_run):
        #bs = max(self.batch_size // 2, 1)
        bs = self.batch_size
        torch.manual_seed(self.seed)
        model = SSFlatModel(num_emotions=len(self.label_list)).to(self.device)
        save_dir = self.models_dir + f"/Run_Nr_{current_run}/flat/"
        return self.run_model_test(lr, epochs, model, save_dir, bs=bs)


    def run_model_test(self, lr, epochs, model, save_dir, bs = None):
        trainDS, testDs = train_val_dataset(self.dataset, val_split=0.2, seed=100)
        valDs, testDs = train_val_dataset(testDs, val_split=0.5, seed=100)
        trainer = SSGenModelTrainer(lr=lr, num_epochs=epochs, model=model, train_dataset=trainDS,
                                    eval_dataset=testDs,
                                    device=self.device, labelList=self.label_list,
                                    batch_size=self.batch_size if bs is None else bs,
                                    save_model_every=self.safe_model_every,
                                    save_highest_acc_min_acc=self.save_highest_acc_min_acc,
                                    model_path = save_dir, regularize_dims=self.regularize_dims)
        return trainer.train()
    
    