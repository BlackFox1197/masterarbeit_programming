import os
import sys

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


def load_custom_dataset():
    paths = []
    testpaths = []
    testlabels = []
    terminator = 'D:/Uni/19.Master/Daten/terminator.wav'
    print(sys.executable)
    emotions = []
    # for dirname, _, filenames in os.walk('Daten/TESS Toronto emotional speech set data'):
    # D:\Uni\19.Master\DATEN
    for dirname, _, filenames in os.walk('../tess'):
        for filename in filenames:
            label = filename.split('_')[-1]
            label = label.split('.')[0]
            if (label != 'neutral'):
                emotions.append(label.lower())
                paths.append(os.path.join(dirname, filename))
    for dirname, _, filenames in os.walk('../Stimuli_Intensitätsmorphs'):
        for filename in filenames:

            intens = filename.split('_')[-2]
            emot = filename.split('_')[1]
            label = emot
            match label:
                case 'ang':
                    label = 'angry'
                case 'dis':
                    label = 'disgust'
                case 'fea':
                    label = 'fear'
                case 'hap':
                    label = 'happy'
                case 'sad':
                    label = 'sad'
                case 'sur':
                    label = 'ps'
            if (emot != 'ple'):
                testpaths.append(os.path.join(dirname, filename))
                testlabels.append(label.lower())
    com_labels = testlabels + emotions
    com_paths = testpaths + paths
    print(testlabels)
    print(testpaths)
    print('Dataset is loaded')
    return paths, emotions, testpaths, testlabels