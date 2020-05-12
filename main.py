import loader
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint


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


def conv_layer(inputs, filters, kernel_size, pool_size):
    x = Conv1D(filters, kernel_size)(inputs)
    x = tf.nn.relu(x)
    x = MaxPooling1D(pool_size=pool_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Dropout(0.4)(x)

    return x


class Classifier:

    @staticmethod
    def build(inputs, classes, layers, filters, kernel_size, pool_size):
        # Convolutional layers
        x = inputs
        for i in range(layers):
            x = conv_layer(x, filters[i], kernel_size[i], pool_size[i])
        x = tf.keras.layers.LSTM(96, return_sequences=False)(x)
        x = Dropout(0.4)(x)
        # Final layer
        x = Flatten()(x)
        x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(), activation='relu')(x)
        x = Dropout(rate=0.4, trainable=True)(x)
        outputs = Dense(classes, activation='softmax')(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs)


def predict(model, x, y):
    prediction = model.predict(x)
    predicted_index = np.argmax(prediction, axis=1)
    print("Target: {}, Predicted label: {}".format(y, predicted_index))


def main():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = loader.get_train_val_test(mode='spectrogram')
    print("There are the following classes:")
    classes = set(y_train.tolist()) & set(y_val.tolist()) & set(y_test.tolist())
    print(classes)

    # Remove small samples
    x_train = np.delete(x_train, [3495, 3496, 3497])
    y_train = np.delete(y_train, [3495, 3496, 3497])
    x_train = np.rollaxis(np.dstack(x_train), -1)
    x_val = np.rollaxis(np.dstack(x_val), -1)
    x_test = np.rollaxis(np.dstack(x_test), -1)
    # x_train = np.expand_dims(x_train, axis=3)
    # x_val = np.expand_dims(x_val, axis=3)
    # x_test = np.expand_dims(x_test, axis=3)

    # One-hot encoding of classes
    dict_labels = {'Electronic': 0, 'Experimental': 1, 'Folk': 2, 'Hip-Hop': 3,
                   'Instrumental': 4, 'International': 5, 'Pop': 6, 'Rock': 7}
    y_train = [dict_labels[y_train[i]] for i in range(y_train.shape[0])]
    y_val = [dict_labels[y_val[i]] for i in range(y_val.shape[0])]
    y_test = [dict_labels[y_test[i]] for i in range(y_test.shape[0])]
    y_train = to_categorical(y_train, num_classes=8)
    y_val = to_categorical(y_val, num_classes=8)
    y_test = to_categorical(y_test, num_classes=8)

    # Training parameters
    epochs = 50  # Train for 30 epochs
    lr = 0.001  # Initial learning rate
    batch_size = 16
    tf.random.set_seed(1234)

    inputs = tf.keras.Input(shape=x_train.shape[1:])
    layers = 3
    filters = [56, 56, 56]
    kernel_size = [5, 5, 5]
    pool_size = [2, 2, 2]
    model = Classifier.build(inputs, 8, layers, filters, kernel_size, pool_size)
    print("Summary:")
    print(model.summary())

    opt = Adam(learning_rate=lr)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    # Callbacks: early stopping and checkpoint
    early_stopping = EarlyStopping(monitor='val_accuracy', verbose=1,
                                   patience=10,
                                   mode='max',
                                   restore_best_weights=True)
    filepath = "weights.{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint, early_stopping]
    history = model.fit(x_train, y_train, batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        callbacks=callbacks_list,
                        epochs=epochs, verbose=1)

    # model = build_model(x_train.shape[1:])
    # optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    # model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model.summary()
    #
    # history = model.fit(x_train, y_train,
    #                     validation_data=(x_val, y_val),
    #                     batch_size=batch_size, epochs=30)

    plot_history(history)

    # Evaluate the model on the test data using `evaluate`
    print("Evaluating model on test data...")
    results = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("Test loss:", results[0])
    print("Test acc:", results[1])

    # test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    # print('\nTest accuracy:', test_acc)


if __name__ == "__main__":
    main()
