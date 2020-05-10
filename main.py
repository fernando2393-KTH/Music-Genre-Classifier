import loader


def main():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = loader.get_train_val_test(mode='spectrogram')
    print("There are the following classes:")
    classes = set(y_train.tolist()) & set(y_val.tolist()) & set(y_test.tolist())
    print(classes)


if __name__ == "__main__":
    main()
