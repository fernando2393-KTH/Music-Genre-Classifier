import matplotlib.pyplot as plt
import constants as cts
from tensorflow.keras.utils import to_categorical


def plot_history(history):

    fig, axs = plt.subplots(2)
    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")
    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="validation error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")
    plt.show()


def targets_to_categorical(y_train, y_val, y_test):
    y_train = [cts.dict_labels[y_train[i]] for i in range(y_train.shape[0])]
    y_val = [cts.dict_labels[y_val[i]] for i in range(y_val.shape[0])]
    y_test = [cts.dict_labels[y_test[i]] for i in range(y_test.shape[0])]
    y_train = to_categorical(y_train, num_classes=len(cts.dict_labels))
    y_val = to_categorical(y_val, num_classes=len(cts.dict_labels))
    y_test = to_categorical(y_test, num_classes=len(cts.dict_labels))

    return y_train, y_val, y_test
