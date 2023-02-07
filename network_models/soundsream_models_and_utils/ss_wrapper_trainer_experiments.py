import torch

from network_models.soundsream_models_and_utils.ss_encoded_dataset import ss_encoded_dataset_full
from network_models.soundsream_models_and_utils.ss_model_conv import SSConvModel3Sec
from network_models.soundsream_models_and_utils.ss_model_dim_red import SSDimRedModel
from network_models.soundsream_models_and_utils.ss_model_flat import SSFlatModel
from network_models.soundsream_models_and_utils.ss_trainer_gen_models import SSGenModelTrainer
from utils.audio_dataset_utils import train_val_dataset


class ExperimentsTrainer:

        models_dir = "content/soundstream/experiments/"
        dataset: ss_encoded_dataset_full
        trials_per_model_type: int
        epochs_per_model = 1000
        start_lr = 1e-4
        lr_quotient = 2
        batch_size =8
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def train_em(self): # TODO: save model at the end
            for trail in range(self.trials_per_model_type):
                lr = self.start_lr /  (self.lr_quotient ** trail)
                self.run_conv_model_test(lr, self.epochs_per_model, trail)
                self.run_dim_red_model_test(lr, self.epochs_per_model, trail)
                self.run_flat_model_test(lr, self.epochs_per_model, trail)


        def run_conv_model_test(self, lr, epochs, current_run):
            label_list = self.dataset.encoded_dataset.dataSet.label_list
            model = SSConvModel3Sec(num_emotions=len(label_list))
            save_dir = self.models_dir + f"Run_Nr_{current_run}_conv/"
            return self.run_model_test(lr, epochs, model, label_list, save_dir)

        def run_dim_red_model_test(self, lr, epochs, current_run):
            label_list = self.dataset.encoded_dataset.dataSet.label_list
            model = SSDimRedModel(num_emotions=len(label_list))
            save_dir = self.models_dir + f"Run_Nr_{current_run}_dimred/"
            return self.run_model_test(lr, epochs, model, label_list, save_dir)

        def run_flat_model_test(self, lr, epochs, current_run):
            label_list = self.dataset.encoded_dataset.dataSet.label_list
            model = SSFlatModel(num_emotions=len(label_list))
            save_dir = self.models_dir + f"Run_Nr_{current_run}_flat/"
            return self.run_model_test(lr, epochs, model, label_list, save_dir)


        def run_model_test(self, lr, epochs, model, label_list, save_dir):
            trainDS, testDs = train_val_dataset(self.dataset, val_split=0.1, seed=100)
            trainer = SSGenModelTrainer(lr=lr, num_epochs=epochs, model=model, train_dataset=trainDS,
                                        eval_dataset=testDs,
                                        device=self.device, labelList=label_list,
                                        batch_size=self.batch_size, save_model_every=200, save_highest_acc_min_acc=0.6,
                                        model_path = save_dir)
            return trainer.train()