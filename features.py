import pandas as pd
import sklearn
import librosa.display
import librosa
import os
from tqdm import tqdm
import warnings
import constants as cts
import numpy as np


def compute_feature(mode, filepath):
    """
    This method loads the a music track and computes the mfcc.
    :param mode: allows to get either spectrogram or mfcc.
    :param filepath: music track.
    :return mfcc of the data track.
    """
    if mode == 'spectrogram':
        y, sr = librosa.load(filepath, duration=10.5, sr=44100, mono=True, offset=0.5)  # Load 10 seconds
        # (same length for every track)
        stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        mel = librosa.feature.melspectrogram(n_mels=64, sr=sr, S=stft ** 2)
        log_mel = librosa.power_to_db(mel, ref=np.max)

        return log_mel

    y, sr = librosa.load(filepath, duration=28.5, sr=44100, mono=True, offset=0.5)  # Load 28 seconds
    # (same length for every track)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = sklearn.preprocessing.scale(mfcc, axis=1)  # Normalize mfcc to have mean 0 and std 1

    return mfcc


class FeatureComputation:

    @staticmethod
    def preprocessing(mode):
        """
        This method parses the data files, calls compute_mfcc and save them into a .csv file.
        """
        warnings.filterwarnings('ignore')
        folders = os.listdir(cts.DATASETS)
        if '.DS_Store' in folders:  # MacOS file system check
            folders.remove('.DS_Store')
        folders.sort()
        feature_dict = {}
        for foldername in tqdm(folders):
            files = os.listdir(cts.DATASETS + foldername)
            if '.DS_Store' in files:  # MacOS file system check
                files.remove('.DS_Store')
            files.sort()
            for file in files:
                if os.path.isdir(cts.DATASETS + foldername):
                    key = file.strip('0')
                    key = key.replace('.mp3', '')
                    feature = compute_feature(mode, cts.DATASETS + foldername + '/' + file)
                    feature_dict[int(key)] = feature

        df = pd.DataFrame(list(feature_dict.items()), columns=['track', mode]).astype(object)
        if mode == 'spectrogram':
            df.to_pickle(cts.SPECTROGRAM)
        else:
            df.to_pickle(cts.MFCC)

