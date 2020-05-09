import loader as ld


def main():
    loader = ld.Loader()
    mfcc = loader.load_mfcc()
    # tracks = loader.load_tracks()
    # features = loader.load_features()
    # (x_train, y_train), (x_val, y_val), (x_test, y_test) = loader.split_dataset(tracks, features)


if __name__ == "__main__":
    main()
