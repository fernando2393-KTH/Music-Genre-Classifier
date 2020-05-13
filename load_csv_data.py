from loader import Loader
import pandas as pd

def load_features(filepath):
    return pd.read_pickle(filepath)

def get_train_val_test(filepath):
    """
    :return training, validation and test datasets.
    """
    loader = Loader()
    mfcc_ = load_features(filepath)  # Load the features dataframe of the dataset songs.
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
    x_train = mfcc_train['spectrogram'].to_numpy()
    x_val = mfcc_val['spectrogram'].to_numpy()
    x_test = mfcc_test['spectrogram'].to_numpy()
    # Convert the target values to numpy arrays.
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)