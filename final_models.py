import utils
import load_csv_data
import numpy as np
import constants as cts
import Models.c_rnn as c_rnn
import Models.cnn_model as cnn_model
import Models.cnn_rnn as cnn_rnn
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_performance(model, x_test, y_test_one_hot, y_test):
    # Evaluate the model on the test data using `evaluate`
    print("Evaluating model on test data...")
    results = model.evaluate(x_test, y_test_one_hot)
    print("Test loss:", results[0])
    print("Test acc:", results[1])

    y_test_pred = np.argmax(model.predict(x_test), axis=1)

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("Classification reprot:")
    print(classification_report(y_test, y_test_pred, digits=8))


def main():
    # Load original dataset
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = \
        load_csv_data.get_train_val_test("Datasets/spectrogram.csv")
    # Load augmented dataset (pitch shift, n_steps=2)
    (x_train_aug_2, y_train_aug_2), (x_val_aug_2, y_val_aug_2), (x_test_aug_2, y_test_aug_2) = \
        load_csv_data.get_train_val_test("Datasets/spectrogram_augment_2.csv")
    # Load augmented dataset (time stretch, rate=0.9)
    (x_train_aug_3, y_train_aug_3), (x_val_aug_3, y_val_aug_3), (x_test_aug_3, y_test_aug_3) = \
        load_csv_data.get_train_val_test("Datasets/spectrogram_augment_3.csv")
    # Load augmented dataset (different 3s of tracks sampled)
    (x_train_aug_4, y_train_aug_4), (x_val_aug_4, y_val_aug_4), (x_test_aug_4, y_test_aug_4) = \
        load_csv_data.get_train_val_test("Datasets/spectrogram_augment_4.csv")
    # Load augmented dataset (pitch shift, n_steps=-2)
    (x_train_aug_5, y_train_aug_5), (x_val_aug_5, y_val_aug_5), (x_test_aug_5, y_test_aug_5) = \
        load_csv_data.get_train_val_test("Datasets/spectrogram_augment_5.csv")
    # Load augmented dataset (different 3s, time stretch, rate=1.1)
    (x_train_aug_6, y_train_aug_6), (x_val_aug_6, y_val_aug_6), (x_test_aug_6, y_test_aug_6) = \
        load_csv_data.get_train_val_test("Datasets/spectrogram_augment_6.csv")

    # Stack datasets
    x_train = np.hstack((x_train, x_train_aug_2, x_train_aug_3, x_train_aug_4, x_train_aug_5, x_train_aug_6))
    y_train = np.hstack((y_train, y_train_aug_2, y_train_aug_3, y_train_aug_4, y_train_aug_5, y_train_aug_6))
    # x_val = np.hstack((x_val, x_val_aug_4))
    # x_test = np.hstack((x_test, x_test_aug_4))
    # y_val = np.hstack((y_val, y_val_aug_4))
    # y_test = np.hstack((y_test, y_test_aug_4))

    x_train = np.rollaxis(np.dstack(x_train), -1)
    x_val = np.rollaxis(np.dstack(x_val), -1)
    x_test = np.rollaxis(np.dstack(x_test), -1)
    x_train = np.expand_dims(x_train, axis=3)
    x_val = np.expand_dims(x_val, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    # Convert labels to one-hot categorical encoding
    y_train, y_val, y_test_one_hot = utils.targets_to_categorical(y_train, y_val, y_test)
    y_test = [cts.dict_labels[y_test[i]] for i in range(y_test.shape[0])]

    # -------Run cnn model------- #
    lr = 2e-4
    epochs = 30
    batch_size = 16
    model = cnn_model.cnn_model((128, 128, 1))
    opt = Adam(lr=lr, decay=lr / epochs)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    # model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

    # Callbacks: early stopping and checkpoint
    early_stopping = EarlyStopping(monitor='val_accuracy', verbose=1,
                                   patience=7,
                                   mode='max',
                                   restore_best_weights=True)

    filepath = "/Models/cnn_model/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max')
    callbacks_list = [early_stopping, checkpoint]
    history = model.fit(x_train, y_train, batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        steps_per_epoch=len(x_train) // batch_size,
                        callbacks=callbacks_list,
                        epochs=epochs, verbose=1)

    utils.plot_history(history)

    # Evaluate the model on the test data using `evaluate`
    print("Evaluating model on test data...")
    results = model.evaluate(x_test, y_test_one_hot, batch_size=batch_size)
    print("Test loss:", results[0])
    print("Test acc:", results[1])

    y_test_pred = np.argmax(model.predict(x_test), axis=1)

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("Classification reprot:")
    print(classification_report(y_test, y_test_pred, digits=8))

    # -------Run cnn model + rnn------- #
    model = cnn_rnn.build_model_cnn_rnn((128, 128, 1))
    checkpoint_callback = ModelCheckpoint("/Models/cnn_rnn/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5",
                                          monitor='val_accuracy', verbose=1,
                                          save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_accuracy', verbose=1,
                                   patience=20,
                                   mode='max',
                                   restore_best_weights=True)
    callbacks_list = [checkpoint_callback, early_stopping]
    history = model.fit(x_train, y_train, batch_size=32, epochs=50,
                        validation_data=(x_val, y_val), verbose=1, callbacks=callbacks_list)

    utils.plot_history(history)
    evaluate_performance(model, x_test, y_test_one_hot, y_test)

    # Evaluate the model on the test data using `evaluate`
    print("Evaluating model on test data...")
    results = model.evaluate(x_test, y_test_one_hot)
    print("Test loss:", results[0])
    print("Test acc:", results[1])

    y_test_pred = np.argmax(model.predict(x_test), axis=1)

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("Classification reprot:")
    print(classification_report(y_test, y_test_pred, digits=8))

    # -------Run c-rnn model------- #

    model = c_rnn.build_model_c_rnn((128, 128, 1))
    checkpoint_callback = ModelCheckpoint("/Models/c_rnn/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5",
                                          monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_accuracy', verbose=1,
                                   patience=10,
                                   mode='max',
                                   restore_best_weights=True)
    callbacks_list = [checkpoint_callback, early_stopping]
    history = model.fit(x_train, y_train, batch_size=32, epochs=30,
                        validation_data=(x_val, y_val), verbose=1, callbacks=callbacks_list)

    utils.plot_history(history)
    evaluate_performance(model, x_test, y_test_one_hot, y_test)


if __name__ == "__main__":
    main()
