"""
This class is in charge of loading the relevant data and splitting the dataset.
note: Check https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/usage.ipynb for a deeper understanding of
      the data format in the .csv files.
"""

import pandas as pd
import mfcc
import constants as cts
from pathlib import Path


class Loader:
    def __init__(self):
        self.features = ['mfcc', 'chroma_cens', 'tonnetz', 'spectral_contrast',
                         ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff'],
                         ['rmse', 'zcr']]  # Main categories of the stored features

    @staticmethod
    def load_features():
        """
        This method loads the data features from the .csv file.
        :return features of the data.
        """
        features = pd.read_csv(cts.METADATA + "features.csv", index_col=0, header=[0, 1, 2])

        return features

    @staticmethod
    def load_echonest():
        """
        This method loads the data echonest features from the .csv file.
        :return echonest features of the data.
        """
        echonest = pd.read_csv(cts.METADATA + "echonest.csv", index_col=0, header=[0, 1, 2])

        return echonest

    @staticmethod
    def load_genres():
        """
        This method loads the data genres from the .csv file.
        :return genres of the data and top level genres.
        """
        genres = pd.read_csv(cts.METADATA + "genres.csv", index_col=0)
        top_level = genres['top_level'].unique()  # This corresponds to the considered "top-level genres"

        print("There is a total of " + str(genres.shape[0]) + " genres.")
        print("There is a total of " + str(len(top_level)) + " top-level genres.")

        return genres, top_level

    @staticmethod
    def load_tracks():
        """
        This method loads the data tracks from the .csv file.
        :return music tracks.
        """
        tracks = pd.read_csv(cts.METADATA + "tracks.csv", index_col=0, header=[0, 1])

        return tracks

    @staticmethod
    def load_mfcc():
        if not Path(cts.MFCC).is_file():  # Check if the file exists
            compute_mfcc = mfcc.MfccComputation()
            compute_mfcc.preprocessing()  # If the file does not exist, create it
        mfcc_val = pd.read_pickle(cts.MFCC)

        return mfcc_val

    @staticmethod
    def get_targets(tracks):
        """
        This methods separates the tracks into dataset by means of the 'cat' feature.
        :param tracks: the music tracks loaded in the format returned by 'load_tracks'.
        :return training set, validation set and test set targets and tracks ID.
        """
        train = tracks['set', 'split'] == 'training'  # Training songs
        val = tracks['set', 'split'] == 'validation'  # Validation songs
        test = tracks['set', 'split'] == 'test'  # Test songs

        y_train = tracks.loc[train, ('track', 'genre_top')]
        y_val = tracks.loc[val, ('track', 'genre_top')]
        y_test = tracks.loc[test, ('track', 'genre_top')]

        return y_train, y_val, y_test


def get_train_val_test():
    """
    :return training, validation and test datasets.
    """
    loader = Loader()
    mfcc_ = loader.load_mfcc()  # Load the mfcc datagrame of the dataset songs.
    tracks = loader.load_tracks()  # Load all the tracks of the big dataset.
    y_train, y_val, y_test = loader.get_targets(tracks)  # Load the target values of all the tracks.
    # Get training mfcc and labels dataframes.
    mfcc_train = mfcc_.loc[mfcc_['track'].isin(y_train.index[:].tolist())]
    y_train = y_train[mfcc_train['track'].to_numpy()]
    y_train = y_train[y_train.notna()]
    mfcc_train = mfcc_train.loc[mfcc_train['track'].isin(y_train.index[:].tolist())]
    # Get validation mfcc and labels dataframes.
    mfcc_val = mfcc_.loc[mfcc_['track'].isin(y_val.index[:].tolist())]
    y_val = y_val[mfcc_val['track'].to_numpy()]
    y_val = y_val[y_val.notna()]
    mfcc_val = mfcc_val.loc[mfcc_val['track'].isin(y_val.index[:].tolist())]
    # Get testing mfcc and labels dataframes.
    mfcc_test = mfcc_.loc[mfcc_['track'].isin(y_test.index[:].tolist())]
    y_test = y_test[mfcc_test['track'].to_numpy()]
    y_test = y_test[y_test.notna()]
    mfcc_test = mfcc_test.loc[mfcc_test['track'].isin(y_test.index[:].tolist())]
    # Get the mfcc values and convert them to numpy arrays.
    x_train = mfcc_train['mfcc'].to_numpy()
    x_val = mfcc_val['mfcc'].to_numpy()
    x_test = mfcc_test['mfcc'].to_numpy()
    # Conver the target values to numpy arrays.
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
