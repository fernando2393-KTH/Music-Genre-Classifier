import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


def build_model_cnn_rnn(input_shape):
    x_input = layers.Input(input_shape)

    x = layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', kernel_initializer='he_uniform')(x_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x1 = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform')(x)
    x1 = layers.BatchNormalization()(x1)

    b1 = layers.MaxPool2D((2, 2))(x)
    b1 = layers.Conv2D(16, (1, 1), padding='same', kernel_initializer='he_uniform')(b1)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Activation('relu')(b1)
    b1 = layers.Conv2D(128, (1, 1), padding='same', kernel_initializer='he_uniform')(b1)
    b1 = layers.BatchNormalization()(b1)

    x = layers.Add()([x1, b1])
    x = layers.Activation('relu')(x)

    x1 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x1 = layers.BatchNormalization()(x1)

    b1 = layers.Conv2D(16, (1, 1), padding='same', kernel_initializer='he_uniform')(b1)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Activation('relu')(b1)
    b1 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_uniform')(b1)
    b1 = layers.BatchNormalization()(b1)

    x = layers.Add()([x1, b1])
    x = layers.Activation('relu')(x)

    x2 = layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(x)
    x2 = layers.BatchNormalization()(x2)

    b2 = layers.MaxPool2D((2, 2))(x)
    b2 = layers.Conv2D(32, (1, 1), padding='same', kernel_initializer='he_uniform')(b2)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Activation('relu')(b2)
    b2 = layers.Conv2D(512, (1, 1), padding='same', kernel_initializer='he_uniform')(b2)
    b2 = layers.BatchNormalization()(b2)

    x = layers.Add()([x2, b2])
    x = layers.Activation('relu')(x)

    x2 = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x2 = layers.BatchNormalization()(x2)

    b2 = layers.Conv2D(32, (1, 1), padding='same', kernel_initializer='he_uniform')(b2)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Activation('relu')(b2)
    b2 = layers.Conv2D(512, (1, 1), padding='same', kernel_initializer='he_uniform')(b2)
    b2 = layers.BatchNormalization()(b2)

    x = layers.Add()([x2, b2])
    x = layers.Activation('relu')(x)

    x3 = layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(x)
    x3 = layers.BatchNormalization()(x3)

    b3 = layers.MaxPool2D((2, 2))(x)
    b3 = layers.Conv2D(64, (1, 1), padding='same', kernel_initializer='he_uniform')(b3)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.Activation('relu')(b3)
    b3 = layers.Conv2D(512, (1, 1), padding='same', kernel_initializer='he_uniform')(b3)
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

    # -------Recurrent Block------- #

    # Pooling layer
    r = layers.MaxPooling2D((1, 2), strides=(1, 2), name='pool_lstm')(x_input)

    # Embedding layer
    squeezed = layers.Lambda(lambda var: tf.keras.backend.squeeze(var, axis=-1))(r)
    # flatten2 = K.squeeze(pool_lstm1, axis = -1)
    # dense1 = layers.Dense(dense_size1)(flatten)

    # Bidirectional GRU
    lstm = layers.Bidirectional(layers.GRU(64))(squeezed)

    # -------Concatenate blocks------- #
    # Concat Output
    concat = layers.concatenate([x, lstm], axis=-1, name='concat')
    # Softmax Output
    output = layers.Dense(8, activation='softmax', name='preds')(concat)
    model_output = output
    model = tf.keras.models.Model(x_input, model_output)
    opt = Adam(lr=0.001)
    # opt = tf.keras.optimizers.RMSprop(lr=0.0005)  # Optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    print(model.summary())

    return model
