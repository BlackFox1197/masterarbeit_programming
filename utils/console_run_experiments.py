import sys
from pathlib import Path
module_path = str(Path.cwd().parents[0] / "network_models/soundstream_models_and_utils")
if module_path not in sys.path:
    sys.path.append(module_path)

sys.path.insert(0, "/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject")
from network_models.soundsream_models_and_utils.ss_wrapper_trainer_experiments import ExperimentsTrainer
import torch
from network_models.soundsream_models_and_utils.ss_encoded_dataset import ss_encoded_dataset_full


device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8
models_dir = "/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks/content/soundstream/experiments/"
trials_per_model_type = 3
epochs_per_model = 20
start_lr = 1e-4
lr_quotient = 2

data_set= ss_encoded_dataset_full(
    csvPath="/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks/content/data/allEncodings.pkl", device="cuda")

exp_trainer = ExperimentsTrainer(dataset=data_set, device=device, models_dir=models_dir, batch_size=batch_size, trials_per_model_type=trials_per_model_type,
                   epochs_per_model=epochs_per_model, start_lr=start_lr, lr_quotient=lr_quotient)
exp_trainer.train_em()