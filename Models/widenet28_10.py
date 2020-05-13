import tensorflow as tf
import numpy as np


def wide_basic(inputs, in_planes, out_planes, stride):
    if stride != 1 or in_planes != out_planes:
        skip_c = tf.keras.layers.Conv2D(out_planes, kernel_size=1, strides=stride, use_bias=True,
                                        padding='same')(inputs)
    else:
        skip_c = inputs

    x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, center=True, trainable=True)(inputs)  # Original
    # implementation had decay. Changed for momentum.
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv2D(out_planes, kernel_size=3, strides=1, use_bias=True, padding='same')(x)
    x = tf.keras.layers.Dropout(rate=0.1, trainable=True)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, center=True,
                                           trainable=True)(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv2D(out_planes, kernel_size=3, strides=stride, use_bias=True, padding='same')(x)

    x = tf.add(skip_c, x)

    return x


def wide_layer(out, in_planes, out_planes, num_blocks, stride):
    strides = [stride] + [1] * int(num_blocks - 1)
    i = 0
    for strid in strides:
        out = wide_basic(out, in_planes, out_planes, strid)
        in_planes = out_planes
        i += 1

    return out


class Classifier:

    @staticmethod
    def make_resnet_filter(inputs, depth=28, widen_factor=10, num_classes=8):
        n = (depth - 4) / 6
        k = widen_factor
        print('| Wide-Resnet %dx%d' % (depth, k))
        nstages = [16, 16 * k, 32 * k, 64 * k]
        x = tf.keras.layers.Conv2D(nstages[0], kernel_size=3, strides=1, use_bias=True, padding='same')(inputs)
        x = wide_layer(x, nstages[0], nstages[1], n, stride=1)
        x = wide_layer(x, nstages[1], nstages[2], n, stride=2)
        x = wide_layer(x, nstages[2], nstages[3], n, stride=2)
        x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, center=True, trainable=True)(x)
        x = tf.nn.relu(x)
        x = tf.keras.layers.AvgPool2D([8, 8])(x)
        x = tf.reshape(x, (-1, 640))
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    @staticmethod
    def data_format(x_train, x_val, x_test):
        x_train = np.expand_dims(np.rollaxis(np.dstack(x_train), -1), axis=3)
        x_val = np.expand_dims(np.rollaxis(np.dstack(x_val), -1), axis=3)
        x_test = np.expand_dims(np.rollaxis(np.dstack(x_test), -1), axis=3)

        return x_train, x_val, x_test
