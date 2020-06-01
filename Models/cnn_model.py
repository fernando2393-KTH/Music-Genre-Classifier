import tensorflow as tf
from tensorflow.keras import layers


def cnn_model(input_shape):
    x_input = layers.Input(input_shape)

    x = layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', kernel_initializer='he_uniform')(x_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x1 = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform')(x)
    x1 = layers.BatchNormalization()(x1)

    b1 = layers.MaxPool2D((2, 2))(x)
    b1 = layers.Conv2D(16, (1, 1), padding='same', kernel_initializer='he_uniform')(b1)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Activation('relu')(b1)
    b1 = layers.Conv2D(64, (1, 1), padding='same', kernel_initializer='he_uniform')(b1)
    b1 = layers.BatchNormalization()(b1)

    x = layers.Add()([x1, b1])
    x = layers.Activation('relu')(x)

    x1 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x1 = layers.BatchNormalization()(x1)

    b1 = layers.Conv2D(16, (1, 1), padding='same', kernel_initializer='he_uniform')(b1)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Activation('relu')(b1)
    b1 = layers.Conv2D(128, (1, 1), padding='same', kernel_initializer='he_uniform')(b1)
    b1 = layers.BatchNormalization()(b1)

    x = layers.Add()([x1, b1])
    x = layers.Activation('relu')(x)

    x2 = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(x)
    x2 = layers.BatchNormalization()(x2)

    b2 = layers.MaxPool2D((2, 2))(x)
    b2 = layers.Conv2D(32, (1, 1), padding='same', kernel_initializer='he_uniform')(b2)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Activation('relu')(b2)
    b2 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_uniform')(b2)
    b2 = layers.BatchNormalization()(b2)

    x = layers.Add()([x2, b2])
    x = layers.Activation('relu')(x)

    x2 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x2 = layers.BatchNormalization()(x2)

    b2 = layers.Conv2D(32, (1, 1), padding='same', kernel_initializer='he_uniform')(b2)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Activation('relu')(b2)
    b2 = layers.Conv2D(128, (1, 1), padding='same', kernel_initializer='he_uniform')(b2)
    b2 = layers.BatchNormalization()(b2)

    x = layers.Add()([x2, b2])
    x = layers.Activation('relu')(x)

    x3 = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(x)
    x3 = layers.BatchNormalization()(x3)

    b3 = layers.MaxPool2D((2, 2))(x)
    b3 = layers.Conv2D(64, (1, 1), padding='same', kernel_initializer='he_uniform')(b3)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.Activation('relu')(b3)
    b3 = layers.Conv2D(64, (1, 1), padding='same', kernel_initializer='he_uniform')(b3)
    b3 = layers.BatchNormalization()(b3)

    x = layers.Add()([x3, b3])
    x = layers.Activation('relu')(x)

    x3 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x3 = layers.BatchNormalization()(x3)

    b3 = layers.Conv2D(64, (1, 1), padding='same', kernel_initializer='he_uniform')(b3)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.Activation('relu')(b3)
    b3 = layers.Conv2D(128, (1, 1), padding='same', kernel_initializer='he_uniform')(b3)
    b3 = layers.BatchNormalization()(b3)

    x = layers.Add()([x3, b3])
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(8, kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('softmax')(x)

    model = tf.keras.models.Model(inputs=x_input, outputs=x, name='model_8')
    model.summary()

    return model
