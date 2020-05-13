import pandas as pd
import sklearn
import librosa.display
import librosa
import os
from tqdm import tqdm
import warnings
import constants as cts


def compute_mfcc(filepath):
    """
    This method loads the a music track and computes the mfcc.
    :param filepath: music track.
    :return mfcc of the data track.
    """
<<<<<<< Updated upstream:mfcc.py
=======
    if mode == 'spectrogram':
        y, sr = librosa.load(filepath, duration=3.5, mono=True, offset=15.5)  # Load 10 seconds
        stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=604))
        mel = librosa.feature.melspectrogram(n_mels=128, sr=sr, S=stft ** 2)
        log_mel = librosa.power_to_db(mel, ref=np.max)

        return log_mel

>>>>>>> Stashed changes:features.py
    y, sr = librosa.load(filepath, duration=28.5, sr=44100, mono=True, offset=0.5)  # Load 28 seconds
    # (same length for every track)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = sklearn.preprocessing.scale(mfcc, axis=1)  # Normalize mfcc to have mean 0 and std 1

    return mfcc

def augment_feature(filepath, rate=0.8, n_steps=2, effect="time_stretch"):
    """
    This method loads the a music track and augments it.
    :param mode: allows to get either spectrogram or mfcc.
    :param filepath: music track.
    :return mfcc of the data track.
    """
    y, sr = librosa.load(filepath, duration=3.5, mono=True, offset=0.5)
    
    if effect == "time_stretch":
        y_changed = librosa.effects.time_stretch(y, rate=rate)
    if effect == "pitch_shift":
        y_changed = librosa.effects.pitch_shift(y, sr, n_steps=n_steps)
        
    stft = np.abs(librosa.stft(y_changed, n_fft=2048, hop_length=604))
    mel = librosa.feature.melspectrogram(n_mels=128, sr=sr, S=stft ** 2)
    log_mel = librosa.power_to_db(mel, ref=np.max)
        
    return log_mel


class MfccComputation:

    @staticmethod
    def preprocessing():
        """
        This method parses the data files, calls compute_mfcc and save them into a .csv file.
        """
        warnings.filterwarnings('ignore')
        folders = os.listdir(cts.DATASETS)
        if '.DS_Store' in folders:  # MacOS file system check
            folders.remove('.DS_Store')
        folders.sort()
<<<<<<< Updated upstream:mfcc.py
        mfcc_dict = {}
=======
        feature_dict = {}
        augment_dict = {}
>>>>>>> Stashed changes:features.py
        for foldername in tqdm(folders):
            files = os.listdir(cts.DATASETS + foldername)
            if '.DS_Store' in files:  # MacOS file system check
                files.remove('.DS_Store')
            files.sort()
            for file in files:
                if os.path.isdir(cts.DATASETS + foldername):
                    key = file.strip('0')
                    key = key.replace('.mp3', '')
<<<<<<< Updated upstream:mfcc.py
                    mfcc = compute_mfcc(cts.DATASETS + foldername + '/' + file)
                    mfcc_dict[int(key)] = mfcc

        df = pd.DataFrame(list(mfcc_dict.items()), columns=['track', 'mfcc']).astype(object)
        df.to_pickle(cts.MFCC)
=======
                    if cts.AUGMENT:
                        aug_feature = augment_feature(cts.DATASETS + foldername + '/' + file, rate=1.1, effect="time_stretch")
                        augment_dict[int(key)] = aug_feature
                    else:
                        feature = compute_feature(mode, cts.DATASETS + foldername + '/' + file)
                        feature_dict[int(key)] = feature

        if cts.AUGMENT:
            df = pd.DataFrame(list(augment_dict.items()), columns=['track', mode]).astype(object)
            df.to_pickle(cts.SPECTROGRAM_AUGMENT)
        else: 
            df = pd.DataFrame(list(feature_dict.items()), columns=['track', mode]).astype(object)
            if mode == 'spectrogram':
                df.to_pickle(cts.SPECTROGRAM)
            else:
                df.to_pickle(cts.MFCC)
>>>>>>> Stashed changes:features.py

