import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


def build_model_c_rnn(model_input):
    x_input = layers.Input(model_input)

    # -------Convolutional blocks------- #
    # First convolutional block
    x = layers.Conv2D(68, (3, 3), strides=(1, 1), padding='same', name='conv_1')(x_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2), strides=(1, 1))(x)
    x = layers.Dropout(0.1)(x)

    # Second convolutional block
    x = layers.Conv2D(137, (3, 3), strides=(1, 1), padding='same', name='conv_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(1, 1))(x)
    x = layers.Dropout(0.1)(x)

    # Third convolutional block
    x = layers.Conv2D(137, (3, 3), strides=(1, 1), padding='same', name='conv_3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((4, 4), strides=(1, 1))(x)
    x = layers.Dropout(0.1)(x)

    # Fourth convolutional block
    x = layers.Conv2D(137, (3, 3), strides=(1, 1), padding='same', name='conv_4')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((4, 4), strides=(1, 1))(x)
    x = layers.Dropout(0.1)(x)

    # -------Recurrent Block------- #
    # GRU layer
    lstm = layers.GRU(68)(x[:, :, :, 0])

    # Softmax Output
    output = layers.Dense(8, activation='softmax', name='preds')(lstm)
    model_output = output
    model = tf.keras.models.Model(x_input, model_output)
    opt = Adam(lr=0.001)
    # opt = tf.keras.optimizers.RMSprop(lr=0.0005)  # Optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    print(model.summary())

    return model
