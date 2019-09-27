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


class DenseEncoder(tf.keras.Model):
    def __init__(self):
        super(DenseEncoder, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(16, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(16, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(DROPOUT_PROB)

    def call(self, inputs, training=None):
        x = self.dropout(self.dense1(inputs), training=training)
        x = self.dropout(self.dense2(x), training=training)
        x = self.dropout(self.dense3(x), training=training)
        return x


class CNNEncoder(tf.keras.Model):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = kl.Conv2D(8, 3, 1, activation=tf.nn.leaky_relu, data_format='channels_last')
        self.conv2 = kl.Conv2D(8, 3, 1, activation=tf.nn.leaky_relu, data_format='channels_last')
        self.conv3 = kl.Conv2D(8, 3, 1, activation=tf.nn.leaky_relu, data_format='channels_last')
        self.flatten = kl.Flatten()
        self.dropout = kl.Dropout(DROPOUT_PROB)

    def call(self, inputs, training=None):
        x = tf.dtypes.cast(inputs, tf.float32)
        x = self.dropout(self.conv1(x), training=training)
        x = self.dropout(self.conv2(x), training=training)
        x = self.dropout(self.conv3(x), training=training)
        return self.flatten(x)


class LSTMConvACCell(tf.keras.layers.Layer):
    def __init__(self, action_shape, Encoder, **kwargs):
        super(LSTMConvACCell, self).__init__(**kwargs)
        self.encoder = Encoder()
        self.dense = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        self.dropout = tf.keras.layers.Dropout(DROPOUT_PROB)

        self.lstm = tf.keras.layers.LSTMCell(17)

        self.value_layer = kl.Dense(1, activation=None, name="value")
        self.action_layer = kl.Dense(np.prod(action_shape), activation=None, name="action")

    @property
    def state_size(self):
        return self.lstm.state_size

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.lstm.get_initial_state(inputs=inputs,
                                           batch_size=batch_size,
                                           dtype=tf.float32)

    def call(self, inputs, states, training=None):
        x = self.encoder(inputs, training=training)
        x = self.dropout(self.dense(x), training=training)
        x, hc_ = self.lstm(x, states, training=training)

        action_logits = self.action_layer(x)
        values_pred = self.value_layer(x)
        return (action_logits, values_pred), hc_

    def action_value(self, observation, states):
        (logits, value), next_hc = self.call(observation, states, training=False)
        action = tf.random.categorical(logits, 1)
        return tf.squeeze(action, axis=-1), tf.squeeze(value, axis=1), next_hc

    def action(self, observation, states):
        (logits, _), _ = self.call(observation, states, training=False)
        action = tf.random.categorical(logits, 1)
        return tf.squeeze(action, axis=-1)



class LSTMAC(tf.keras.Model):

    def __init__(self, action_shape, Encoder):
        super(LSTMAC, self).__init__()
        self.cell = LSTMConvACCell(action_shape, Encoder)
        self.rnn = tf.keras.layers.RNN(self.cell, return_sequences=True)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return self.rnn(inputs, mask=mask, training=training, initial_state=initial_state)



class ConvLSTMAC(tf.keras.Model):
    def __init__(self, action_shape):
        super(ConvLSTMAC, self).__init__()
        self.rnn = kl.ConvLSTM2D(8, (3, 3), 1, "same")




