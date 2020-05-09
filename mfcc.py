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
    y, sr = librosa.load(filepath, duration=28.5, sr=44100, mono=True, offset=0.5)  # Load 28 seconds
    # (same length for every track)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = sklearn.preprocessing.scale(mfcc, axis=1)  # Normalize mfcc to have mean 0 and std 1

    return mfcc


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
        mfcc_dict = {}
        for foldername in tqdm(folders):
            files = os.listdir(cts.DATASETS + foldername)
            files.sort()
            for file in files:
                if os.path.isdir(cts.DATASETS + foldername):
                    key = file.strip('0')
                    key = key.replace('.mp3', '')
                    mfcc = compute_mfcc(cts.DATASETS + foldername + '/' + file)
                    mfcc_dict[key] = mfcc

        df = pd.DataFrame(list(mfcc_dict.items()), columns=['track', 'mfcc']).astype(object)
        df.to_pickle(cts.MFCC)

