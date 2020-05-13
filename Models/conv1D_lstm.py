import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout


def conv_layer(inputs, filters, kernel_size, pool_size):
    x = Conv1D(filters, kernel_size)(inputs)
    x = tf.nn.relu(x)
    x = MaxPooling1D(pool_size=pool_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Dropout(0.4)(x)

    return x


class Classifier:

    def __init__(self, layers, filters, kernel_size, pool_size):
        self.layers = layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size

    def build(self, inputs, classes):
        # Convolutional layers
        x = inputs
        for i in range(self.layers):
            x = conv_layer(x, self.filters[i], self.kernel_size[i], self.pool_size[i])
        x = tf.keras.layers.LSTM(96, return_sequences=False)(x)
        x = Dropout(0.4)(x)
        # Final layer
        x = Flatten()(x)
        x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(), activation='relu')(x)
        x = Dropout(rate=0.4, trainable=True)(x)
        outputs = Dense(classes, activation='softmax')(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    @staticmethod
    def data_format(x_train, x_val, x_test):
        x_train = np.rollaxis(np.dstack(x_train), -1)
        x_val = np.rollaxis(np.dstack(x_val), -1)
        x_test = np.rollaxis(np.dstack(x_test), -1)

        return x_train, x_val, x_test
