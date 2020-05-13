import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Add, Conv2D, Activation, MaxPool2D, Dense, Flatten, BatchNormalization


class Classifier:

    @staticmethod
    def build(inputs):
        x = Conv2D(32, (5, 5), strides=(2, 2), padding='same', kernel_initializer='he_uniform')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x1 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform')(x)
        x1 = BatchNormalization()(x1)

        b1 = MaxPool2D((2, 2))(x)
        b1 = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_uniform')(b1)
        b1 = BatchNormalization()(b1)
        b1 = Activation('relu')(b1)
        b1 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_uniform')(b1)
        b1 = BatchNormalization()(b1)

        x = Add()([x1, b1])
        x = Activation('relu')(x)

        x1 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
        x1 = BatchNormalization()(x1)

        b1 = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_uniform')(x)
        b1 = BatchNormalization()(b1)
        b1 = Activation('relu')(b1)
        b1 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_uniform')(b1)
        b1 = BatchNormalization()(b1)

        x = Add()([x1, b1])
        x = Activation('relu')(x)

        x2 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(x)
        x2 = BatchNormalization()(x2)

        b2 = MaxPool2D((2, 2))(x)
        b2 = Conv2D(32, (1, 1), padding='same', kernel_initializer='he_uniform')(b2)
        b2 = BatchNormalization()(b2)
        b2 = Activation('relu')(b2)
        b2 = Conv2D(512, (1, 1), padding='same', kernel_initializer='he_uniform')(b2)
        b2 = BatchNormalization()(b2)

        x = Add()([x2, b2])
        x = Activation('relu')(x)

        x2 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x2 = BatchNormalization()(x2)

        b2 = Conv2D(32, (1, 1), padding='same', kernel_initializer='he_uniform')(b2)
        b2 = BatchNormalization()(b2)
        b2 = Activation('relu')(b2)
        b2 = Conv2D(512, (1, 1), padding='same', kernel_initializer='he_uniform')(b2)
        b2 = BatchNormalization()(b2)

        x = Add()([x2, b2])
        x = Activation('relu')(x)

        x3 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(x)
        x3 = BatchNormalization()(x3)

        b3 = MaxPool2D((2, 2))(x)
        b3 = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_uniform')(b3)
        b3 = BatchNormalization()(b3)
        b3 = Activation('relu')(b3)
        b3 = Conv2D(512, (1, 1), padding='same', kernel_initializer='he_uniform')(b3)
        b3 = BatchNormalization()(b3)

        x = Add()([x3, b3])
        x = Activation('relu')(x)

        x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D((2, 2))(x)

        x = Flatten()(x)
        x = Dense(128, kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(8, kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        outputs = Activation('softmax')(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    @staticmethod
    def data_format(x_train, x_val, x_test):
        x_train = np.expand_dims(np.rollaxis(np.dstack(x_train), -1), axis=3)
        x_val = np.expand_dims(np.rollaxis(np.dstack(x_val), -1), axis=3)
        x_test = np.expand_dims(np.rollaxis(np.dstack(x_test), -1), axis=3)

        return x_train, x_val, x_test
