import loader
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm


class Classifier:

    @staticmethod
    def build(inputs, classes):

        x = Conv2D(16, (3, 3), strides=(1, 1))(inputs)
        x = tf.nn.relu(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)

        x = Conv2D(32, (3, 3), strides=(1, 1))(x)
        x = tf.nn.relu(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)

        x = Conv2D(64, (3, 3), strides=(1, 1))(x)
        x = tf.nn.relu(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
        x = Dropout(rate=0.1, trainable=True)(x)

        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(rate=0.3, trainable=True)(x)
        outputs = Dense(classes, activation='softmax')(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs)


def main():
    (x_train_prev, y_train), (x_val_prev, y_val), (x_test_prev, y_test) = loader.get_train_val_test(mode='spectrogram')
    print("There are the following classes:")
    classes = set(y_train.tolist()) & set(y_val.tolist()) & set(y_test.tolist())
    print(classes)

    # Remove small samples
    x_train_prev = np.delete(x_train_prev, [3495, 3496, 3497])
    y_train = np.delete(y_train, [3495, 3496, 3497])

    x_train = np.zeros((x_train_prev.shape[0], x_train_prev[0].shape[0], x_train_prev[0].shape[1]))
    x_val = np.zeros((x_val_prev.shape[0], x_val_prev[0].shape[0], x_val_prev[0].shape[1]))
    x_test = np.zeros((x_test_prev.shape[0], x_test_prev[0].shape[0], x_test_prev[0].shape[1]))
    for i in tqdm(range(x_train_prev.shape[0])):
        x_train[i] = x_train_prev[i]
    del x_train_prev
    for i in tqdm(range(x_val_prev.shape[0])):
        x_val[i] = x_val_prev[i]
    del x_val_prev
    for i in tqdm(range(x_test_prev.shape[0])):
        x_test[i] = x_test_prev[i]
    del x_test_prev

    x_train = np.expand_dims(x_train, axis=3)
    x_val = np.expand_dims(x_val, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

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
    epochs = 25
    lr = 0.005
    batch_size = 16

    inputs = tf.keras.Input(shape=x_train.shape[1:])
    model = Classifier.build(inputs, classes=8)
    print("Summary:")
    print(model.summary())

    opt = Adam(lr=lr, decay=lr / epochs)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    # Callbacks: early stopping and checkpoint
    early_stopping = EarlyStopping(monitor='val_accuracy', verbose=1,
                                   patience=10,
                                   mode='max',
                                   restore_best_weights=True)
    filepath = "weights.{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max')
    callbacks_list = [early_stopping, checkpoint]
    history = model.fit(x_train, y_train, batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        steps_per_epoch=len(x_train) // batch_size,
                        callbacks=callbacks_list,
                        epochs=epochs, verbose=1)

    print("\nHistory dict:", history.history)

    # Evaluate the model on the test data using `evaluate`
    print("Evaluating model on test data...")
    results = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("Test loss:", results[0])
    print("Test acc:", results[1])


if __name__ == "__main__":
    main()
