import sys
from pathlib import Path

module_path = str(Path.cwd().parents[0] / "network_models/soundstream_models_and_utils")
if module_path not in sys.path:
    sys.path.append(module_path)

sys.path.insert(0, "/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject")
import network_models.soundsream_models_and_utils.ss_encoded_dataset as ssed

dataset = ssed.ss_encoded_dataset_full(
    directory_cafe="/home/ckwdani/Music/emotionDatasets/converted_mono/cafe",
    directory_tess="/home/ckwdani/Music/emotionDatasets/converted_mono/tess",
    directory_ravdess="/home/ckwdani/Music/emotionDatasets/converted_mono/RAVDESS Audio_Speech_Actors_01-24",
    directory_mesd="/home/ckwdani/Music/emotionDatasets/converted_mono/mesd",
    device="cuda",
    #sound_stream_path="../notebooks/content/soundstream/vers0.7.4/01_Soundstream_7_x_new_libri_full/5_3250/soundstream.3250.pt")
    #sound_stream_path="/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks/content/soundstream/vers0.7.4/01_Soundstream_7_x_new_libri_full/currentSelection/soundstream.72500.pt")
    sound_stream_path="/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks/content/soundstream/verision0.12.1/30_10000_1e-4_bs6_gae8_dml320-32---MUSIC-Emotion/soundstream.6000.pt",
    clip_path="/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks/content/soundstream/experiments/AUTO_encoder/Nr1/1500_moreencoder_0.pth", # trained for 1000 epochs
    encoder=True,
    circular=True
)
dataset.saveEncoding("../notebooks/content/data/allEncodings_encoder.pkl")