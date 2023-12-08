import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def vit_block(input_tensor, num_heads, transformer_units, sequence_length):
    # Multi-Head Attention
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=transformer_units)(input_tensor, input_tensor)
    x = layers.Add()([x, input_tensor])
    x = layers.LayerNormalization()(x)

    # Feed Forward Network
    y = layers.Dense(transformer_units, activation="relu")(x)
    y = layers.Dense(sequence_length)(y)
    y = layers.Add()([y, x])
    y = layers.LayerNormalization()(y)

    return y

def build_vit(input_shape, patch_size, num_layers, num_heads, transformer_units, num_classes):
    # Calculate the number of patches and the sequence length
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    sequence_length = num_patches

    inputs = keras.Input(shape=input_shape)

    # Create Patches
    x = layers.Conv2D(transformer_units, kernel_size=patch_size, strides=patch_size, padding="valid")(inputs)
    x = layers.Reshape((num_patches, transformer_units))(x)

    # Positional Embedding
    position_embedding = layers.Embedding(input_dim=num_patches, output_dim=transformer_units)
    positions = tf.range(start=0, limit=num_patches, delta=1)
    x = x + position_embedding(positions)

    # Transformer Blocks
    for _ in range(num_layers):
        x = vit_block(x, num_heads, transformer_units, sequence_length)

    # Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
def _res_net_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation="relu",
                      padding="same")(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, conv_size, activation=None, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, input_data])
    x = layers.Activation("relu")(x)

    return x


def build_resnet(input_shape, classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D(3)(x)

    for i in range(10):
        x = _res_net_block(x, 64, 3)

    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    if classes == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"
    outputs = layers.Dense(classes, activation=activation)(x)

    res_net_model = keras.Model(inputs, outputs)
    return res_net_model


def build_simple_cnn(input_shape, classes):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(3, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(8, 3, padding="same", activation="relu")(x)
    x = layers.AveragePooling2D()(x)  # 64

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.AveragePooling2D()(x)  # 32

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)

    x = layers.Flatten()(x)
    if classes == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"
    outputs = layers.Dense(classes, activation=activation)(x)

    model = keras.Model(inputs, outputs)
    return model


def build_simple_nn(input_shape, classes, l2=0.01):
    inputs = keras.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(l2))(x)
    # x = layers.Dense(128, activation='relu',
    #                  kernel_regularizer=keras.regularizers.l2(l2))(x)

    if classes == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"
    outputs = layers.Dense(classes, activation=activation)(x)

    model = keras.Model(inputs, outputs)
    return model


def build_multinomial_regression(input_shape, classes, kernel_regularizer=None, dataset=None):
    inputs = keras.Input(shape=input_shape)
    x = layers.Flatten()(inputs)

    if classes == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"
    outputs = layers.Dense(classes, activation=activation,
                           kernel_regularizer=kernel_regularizer)(x)

    model = keras.Model(inputs, outputs)
    return model


def build_svm(input_shape, classes, l_2, logits=False):
    inputs = keras.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(classes, activation="linear",
                     kernel_regularizer=keras.regularizers.l2(l_2))(x)

    if logits:
        return keras.Model(inputs, x)

    if classes > 1:
        outputs = 2 * layers.Softmax()(2*x) - 1
    else:
        outputs = layers.Activation(tf.nn.tanh)(x)

    model = keras.Model(inputs, outputs)
    return model


def build_multinomial_regression_l2(input_shape, classes, l_2=0.01):
    return build_multinomial_regression(input_shape, classes, kernel_regularizer=keras.regularizers.l2(l_2))


def build_multinomial_regression_l1(input_shape, classes, l_1=0.1):
    return build_multinomial_regression(input_shape, classes, kernel_regularizer=keras.regularizers.l1(l_1))


def build_multinomial_regression_l1_l2(input_shape, classes, l_1=0.01, l_2=0.01):
    return build_multinomial_regression(input_shape, classes, kernel_regularizer=keras.regularizers.l1_l2(l_1, l_2))
