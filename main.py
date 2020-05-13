import loader
import Models.utils as utils
import Models.conv1D_lstm as conv1
import Models.conv2D as conv2
import tensorflow as tf
import Models.widenet28_10 as widenet
from Models.wide_learning_rate import WideLearningRate
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint


def main():
    # Mode: ['mfcc', 'spectrogram']
    # Filename: Include the name of the file stored in 'Datasets/' without the extension.
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = loader.get_train_val_test(mode='spectrogram', filename="")
    print("There are the following categories:")
    categories = set(y_train.tolist()) & set(y_val.tolist()) & set(y_test.tolist())
    print(categories)

    # Instatiate the classifier model
    # classifier = conv1.Classifier(layers=3,
    #                                  filters=[56, 56, 56],
    #                                  kernel_size=[5, 5, 5],
    #                                  pool_size=[2, 2, 2])  # Uncomment for conv1
    # classifier = conv2.Classifier()  # Uncomment for conv2
    classifier = widenet.Classifier()  # Uncomment for widenet
    # Format input for this type of network
    x_train, x_val, x_test = classifier.data_format(x_train, x_val, x_test)
    # One-hot encoding of classes
    y_train, y_val, y_test = utils.targets_to_categorical(categories, y_train, y_test, y_val)
    # Training parameters
    epochs = 50  # Train for 30 epochs
    lr = 0.001  # Initial learning rate for Adam
    batch_size = 16
    tf.random.set_seed(1234)
    # Get the inputs placeholder
    inputs = tf.keras.Input(shape=x_train.shape[1:])
    model = classifier.make_resnet_filter(inputs, depth=28, widen_factor=10)  # Uncomment for widenet
    # model = classifier.build(inputs, len(categories))  # Uncomment for conv1
    # model = classifier.build(inputs)  # Uncomment for conv2
    print("Summary:")
    print(model.summary())
    # opt = Adam(learning_rate=lr)  # Uncomment for conv1 and conv2
    opt = SGD(lr=WideLearningRate.get_learning_rate(epoch=0),
              momentum=0.9, nesterov=True)  # Uncomment for widenet
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    # Callbacks: early stopping and checkpoint
    early_stopping = EarlyStopping(monitor='val_accuracy', verbose=1,
                                   patience=10,
                                   mode='max',
                                   restore_best_weights=True)
    filepath = "Weights/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint, early_stopping]
    history = model.fit(x_train, y_train, batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        callbacks=callbacks_list,
                        epochs=epochs, verbose=1)

    utils.plot_history(history)

    # Evaluate the model on the test data using `evaluate`
    print("Evaluating model on test data...")
    results = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("Test loss:", results[0])
    print("Test acc:", results[1])


if __name__ == "__main__":
    main()
