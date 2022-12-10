import librosa
import torchaudio

def speech_file_to_array_fn(path, target_sampling_rate):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def label_to_id(label, label_list):

    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label

def speech_file_to_array_librosa(speech_path, target_sampling_rate):
    speech_array, sampling_rate = librosa.load(speech_path)
    resampler = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=target_sampling_rate).squeeze()

    return resampler