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
    #directory_induced="/home/ckwdani/Music/emotionDatasets/converted/induced_emos",
    device="cuda",
    #sound_stream_path="../notebooks/content/soundstream/vers0.7.4/01_Soundstream_7_x_new_libri_full/5_3250/soundstream.3250.pt")
    #sound_stream_path="/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks/content/soundstream/vers0.7.4/01_Soundstream_7_x_new_libri_full/currentSelection/soundstream.72500.pt")
    sound_stream_path="/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks_clip/content/models/soundstream_model/soundstream.44000.pt",
    #clip_path="/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks/content/encoder/clip_full_datastet/encoder_600.pth" # trained for 1500 epochs
    clip_path="/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks_clip/content/models/clip_models/clip_full_dataset_0_2_sec_encoder_800.pth",
    seconds=0.2,
    umap = True,
    umap_dims=3
)
dataset.saveEncoding("/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks_clip/content/datasets/clip_encoded/umap_3_dims/all_encodings_with_umap_0_2_sec.pkl")


dataset = ssed.ss_encoded_dataset_full(
    directory_cafe="/home/ckwdani/Music/emotionDatasets/converted_mono/cafe",
    directory_tess="/home/ckwdani/Music/emotionDatasets/converted_mono/tess",
    directory_ravdess="/home/ckwdani/Music/emotionDatasets/converted_mono/RAVDESS Audio_Speech_Actors_01-24",
    directory_mesd="/home/ckwdani/Music/emotionDatasets/converted_mono/mesd",
    #directory_induced="/home/ckwdani/Music/emotionDatasets/converted/induced_emos",
    device="cuda",
    #sound_stream_path="../notebooks/content/soundstream/vers0.7.4/01_Soundstream_7_x_new_libri_full/5_3250/soundstream.3250.pt")
    #sound_stream_path="/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks/content/soundstream/vers0.7.4/01_Soundstream_7_x_new_libri_full/currentSelection/soundstream.72500.pt")
    sound_stream_path="/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks_clip/content/models/soundstream_model/soundstream.44000.pt",
    #clip_path="/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks/content/encoder/clip_full_datastet/encoder_600.pth" # trained for 1500 epochs
    clip_path="/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks_clip/content/models/clip_models/clip_full_dataset_1_sec_encoder_400.pth",
    seconds=1.0,
    umap = True,
    umap_dims=3
)
dataset.saveEncoding("/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks_clip/content/datasets/clip_encoded/umap_3_dims/all_encodings_with_umap_1_sec.pkl")


dataset = ssed.ss_encoded_dataset_full(
    directory_cafe="/home/ckwdani/Music/emotionDatasets/converted_mono/cafe",
    directory_tess="/home/ckwdani/Music/emotionDatasets/converted_mono/tess",
    directory_ravdess="/home/ckwdani/Music/emotionDatasets/converted_mono/RAVDESS Audio_Speech_Actors_01-24",
    directory_mesd="/home/ckwdani/Music/emotionDatasets/converted_mono/mesd",
    #directory_induced="/home/ckwdani/Music/emotionDatasets/converted/induced_emos",
    device="cuda",
    #sound_stream_path="../notebooks/content/soundstream/vers0.7.4/01_Soundstream_7_x_new_libri_full/5_3250/soundstream.3250.pt")
    #sound_stream_path="/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks/content/soundstream/vers0.7.4/01_Soundstream_7_x_new_libri_full/currentSelection/soundstream.72500.pt")
    sound_stream_path="/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks_clip/content/models/soundstream_model/soundstream.44000.pt",
    #clip_path="/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks/content/encoder/clip_full_datastet/encoder_600.pth" # trained for 1500 epochs
    clip_path="/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks_clip/content/models/clip_models/clip_full_dataset_3_5_sec_encoder_600.pth",
    seconds=3.5,
    umap = True,
    umap_dims=3
)
dataset.saveEncoding("/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks_clip/content/datasets/clip_encoded/umap_3_dims/all_encodings_with_umap_3_5_sec.pkl")

dataset = ssed.ss_encoded_dataset_full(
    directory_cafe="/home/ckwdani/Music/emotionDatasets/converted_mono/cafe",
    directory_tess="/home/ckwdani/Music/emotionDatasets/converted_mono/tess",
    directory_ravdess="/home/ckwdani/Music/emotionDatasets/converted_mono/RAVDESS Audio_Speech_Actors_01-24",
    directory_mesd="/home/ckwdani/Music/emotionDatasets/converted_mono/mesd",
    #directory_induced="/home/ckwdani/Music/emotionDatasets/converted/induced_emos",
    device="cuda",
    #sound_stream_path="../notebooks/content/soundstream/vers0.7.4/01_Soundstream_7_x_new_libri_full/5_3250/soundstream.3250.pt")
    #sound_stream_path="/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks/content/soundstream/vers0.7.4/01_Soundstream_7_x_new_libri_full/currentSelection/soundstream.72500.pt")
    sound_stream_path="/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks_clip/content/models/soundstream_model/soundstream.44000.pt",
    #clip_path="/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks/content/encoder/clip_full_datastet/encoder_600.pth" # trained for 1500 epochs
    clip_path="/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks_clip/content/models/clip_models/clip_full_dataset_5_sec_encoder_200.pth",
    seconds=5.0,
    umap = True,
    umap_dims=3
)
dataset.saveEncoding("/home/ckwdani/Programming/Projects/masterarbeit/Jupyter/mainProject/notebooks_clip/content/datasets/clip_encoded/umap_3_dims/all_encodings_with_umap_5_sec.pkl")