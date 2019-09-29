"""
File: Actor
Date: 2019-07-28 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl

DROPOUT_PROB = 0.057  # yeah super janky


def get_encoder_type(encoder_name):
    """ gets the class of the encoer of the given name """
    if encoder_name == 'Dense':
        return DenseEncoder
    elif encoder_name == 'CNN':
        return CNNEncoder
    else:
        raise ValueError(encoder_name)


class DenseEncoder(tf.keras.layers.Layer):
    def __init__(self, input_shape=None):
        super(DenseEncoder, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, activation=tf.nn.relu, input_shape=input_shape)
        self.dense2 = tf.keras.layers.Dense(16, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(16, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(DROPOUT_PROB)

    def call(self, inputs, training=None):
        x = self.dropout(self.dense1(inputs), training=training)
        x = self.dropout(self.dense2(x), training=training)
        x = self.dropout(self.dense3(x), training=training)
        return x


class CNNEncoder(tf.keras.layers.Layer):
    def __init__(self, input_shape=None):
        super(CNNEncoder, self).__init__()
        self.conv1 = kl.Conv2D(8, 3, 1, activation=tf.nn.leaky_relu, data_format='channels_last', input_shape=input_shape)
        self.conv2 = kl.Conv2D(8, 3, 1, activation=tf.nn.leaky_relu, data_format='channels_last')
        self.conv3 = kl.Conv2D(8, 3, 1, activation=tf.nn.leaky_relu, data_format='channels_last')
        self.flatten = kl.Flatten()
        self.dropout = kl.Dropout(DROPOUT_PROB)

    def compute_output_shape(self, input_shape):
        shape = self.conv1.compute_output_shape(input_shape)
        shape = self.conv2.compute_output_shape(shape)
        shape = self.conv3.compute_output_shape(shape)
        return self.flatten.compute_output_shape(shape)

    def call(self, inputs, training=None):
        x = tf.dtypes.cast(inputs, tf.float32)
        x = self.dropout(self.conv1(x), training=training)
        x = self.dropout(self.conv2(x), training=training)
        x = self.dropout(self.conv3(x), training=training)
        return self.flatten(x)


class LSTMAC(tf.keras.Model):

    def __init__(self, input_shape, action_shape, Encoder, return_sequences=True):
        super(LSTMAC, self).__init__()
        self.encoder = tf.keras.layers.TimeDistributed(Encoder(input_shape=input_shape))
        self.dense = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        self.dropout = tf.keras.layers.Dropout(DROPOUT_PROB)
        self.lstm = tf.keras.layers.LSTM(64, return_sequences=return_sequences)

        self.value_layer = kl.Dense(1, activation=None, name="value")
        self.action_layer = kl.Dense(np.prod(action_shape), activation=None, name="action")

    def call(self, inputs, mask=None, training=None, initial_state=None):
        x = self.encoder(inputs, training=training)
        x = self.dropout(self.dense(x), training=training)
        x = self.lstm(x, mask=mask, training=training, initial_state=initial_state)
        return self.action_layer(x), self.value_layer(x)


class ConvLSTMAC(tf.keras.Model):

    def __init__(self, action_shape):
        super(ConvLSTMAC, self).__init__()
        self.rnn = kl.ConvLSTM2D(8, (3, 3), 1, "same")
        self.flatten = kl.Flatten()
        self.value_layer = kl.Dense(1, activation=None, name="value")
        self.action_layer = kl.Dense(np.prod(action_shape), activation=None, name="action")

    def call(self, input, mask=None, training=None, initial_state=None):
        x = self.rnn(input, mask=mask, training=training, initial_state=initial_state)
        x = self.flatten(x)
        return self.action_layer(x), self.value_layer(x)
