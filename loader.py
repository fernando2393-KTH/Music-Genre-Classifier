"""
This class is in charge of loading the relevant data and splitting the dataset.
note: Check https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/usage.ipynb for a deeper understanding of
      the data format in the .csv files.
"""

import pandas as pd
import features
import constants as cts
from pathlib import Path


class Loader:
    def __init__(self):
        self.features = ['mfcc', 'chroma_cens', 'tonnetz', 'spectral_contrast',
                         ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff'],
                         ['rmse', 'zcr']]  # Main categories of the stored features

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
    def load_features(mode):
        if mode == 'spectrogram':
            if not Path(cts.SPECTROGRAM).is_file():  # Check if the file exists
                compute_ = features.FeatureComputation()
                compute_.preprocessing(mode)  # If the file does not exist, create it
            mfcc_val = pd.read_pickle(cts.SPECTROGRAM)

        else:
            if not Path(cts.MFCC).is_file():  # Check if the file exists
                compute_ = features.FeatureComputation()
                compute_.preprocessing(mode)  # If the file does not exist, create it
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


def process_data(mfcc_, y):
    mfcc = mfcc_.loc[mfcc_['track'].isin(y.index[:].tolist())]
    y = y[mfcc['track'].to_numpy()]
    y = y[y.notna()]
    mfcc_train = mfcc.loc[mfcc['track'].isin(y.index[:].tolist())]

    return mfcc, y


def get_train_val_test(mode='spectrogram'):
    """
    :return training, validation and test datasets.
    """
    loader = Loader()
    print("Calculating " + mode + "...")
    mfcc_ = loader.load_features(mode)  # Load the features dataframe of the dataset songs.
    tracks = loader.load_tracks()  # Load all the tracks of the big dataset.
    y_train, y_val, y_test = loader.get_targets(tracks)  # Load the target values of all the tracks.
    # Get training mfcc and labels dataframes.
    mfcc_train, y_train = process_data(mfcc_, y_train)
    # Get validation mfcc and labels dataframes.
    mfcc_val, y_val = process_data(mfcc_, y_val)
    # Get testing mfcc and labels dataframes.
    mfcc_test, y_test = process_data(mfcc_, y_test)
    # Get the mfcc values and convert them to numpy arrays.
    x_train = mfcc_train[mode].to_numpy()
    x_val = mfcc_val[mode].to_numpy()
    x_test = mfcc_test[mode].to_numpy()
    # Conver the target values to numpy arrays.
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
