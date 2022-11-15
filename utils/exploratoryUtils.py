import math

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def waveplot(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    # librosa.display.waveplot(data, sr=sr)
    # plt.show

def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11, 4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()


def extract_mfcc(filename, duration=3, offset=0.5):
    y, sr = librosa.load(filename, duration=duration, offset=offset)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc


def extract_f0(filename, duration=3, offset=0.5):
    y, sr = librosa.load(filename, duration=duration, offset=offset)
    f0, voiced_flag, voiced_probs = librosa.pyin(y=y, fmin=1, fmax=20000)
    times = librosa.times_like(f0)
    plot_f0(times, f0)
    return times, f0


def extract_f0_andPlot(filename, duration=3, offset=0.5):
    y, sr = librosa.load(filename, duration=duration, offset=offset)
    f0, voiced_flag, voiced_probs = librosa.pyin(y=y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    times = librosa.times_like(f0)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    fig, ax = plt.subplots()

    img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)

    ax.set(title='pYIN fundamental frequency estimation')

    fig.colorbar(img, ax=ax, format="%+2.f dB")

    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)

    ax.legend(loc='upper right')
    return times

def plot_f0(times, f0, label="yin"):
    fig, ax = plt.subplots()
    ax.set(title=label)
    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    ax.legend(loc='upper right')


def average_over_one_emotion_f0(filenameArray, duration=1, offset=0.5, label = "yin"):
    length = len(filenameArray)
    timesAcc = np.zeros((50,))
    f0Acc = np.zeros((50,))
    for filename in filenameArray:
        y, sr = librosa.load(filename, duration=duration, offset=offset)
        f0 = librosa.yin(y=y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        times = librosa.times_like(f0)
        f0Acc = add_two_arrays_different_shape_average(f0Acc,  f0)
        timesAcc = add_two_arrays_different_shape_average(timesAcc, times)
    plot_f0(timesAcc, f0Acc, label)
    return timesAcc, f0Acc


def add_two_arrays_different_shape_average(arr1, arr2):
    big_array = None
    small_array = None


    if(len(arr2) >= len(arr1)):
        big_array = arr2
        small_array = arr1
    else:
        big_array = arr1
        small_array = arr2

    big_array = removeNans(big_array)
    small_array = removeNans(small_array)
    for i in range(0, len(small_array)):
        # if(math.isnan(small_array[i])):
        #     continue
        # if(math.isnan(big_array[i])):
        #     continue
        big_array[i] = (big_array[i] + small_array[i])/2

    return big_array

def removeNans(array):
    if(math.isnan(array[0])):
        array[0] = 0
    for i in range(0, len(array)):
        if(math.isnan(array[i])):
            array[i] = array[i-1]
    return array
























