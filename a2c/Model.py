"""
File: Actor
Date: 2019-07-28 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import tensorflow as tf
import tensorflow.keras.layers as kl

import numpy as np

DROPOUT_PROB = 0.057


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
        self.conv1 = kl.Conv2D(8, 3, 1, activation=tf.nn.leaky_relu)
        self.conv2 = kl.Conv2D(8, 3, 1, activation=tf.nn.leaky_relu)
        self.conv3 = kl.Conv2D(8, 3, 1, activation=tf.nn.leaky_relu)
        self.flatten = kl.Flatten()
        self.dropout = kl.Dropout(DROPOUT_PROB)

    def call(self, inputs, training=None):
        x = tf.dtypes.cast(inputs, tf.float32)
        x = self.dropout(self.conv1(x), training=training)
        x = self.dropout(self.conv2(x), training=training)
        x = self.dropout(self.conv3(x), training=training)
        return self.flatten(x)


class ActorCriticCell(tf.keras.layers.Layer):
    def __init__(self, action_shape, Encoder):
        super(ActorCriticCell, self).__init__()
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

    def call(self, inputs, hc, training=None):
        x = self.encoder(inputs, training=training)
        x = self.dropout(self.dense(x), training=training)
        x, hc_ = self.lstm(x, hc, training=training)

        action_param = self.action_layer(x)
        values = self.value_layer(x)
        return (action_param, values), hc_

    def action_value(self, observation, hc):
        (logits, value), next_hc = self.call(observation, hc, training=False)
        action = tf.random.categorical(logits, 1)
        return tf.squeeze(action, axis=-1), tf.squeeze(value, axis=1), next_hc

    def action(self, observation, hc):
        (logits, _), _ = self.call(observation, hc, training=False)
        action = tf.random.categorical(logits, 1)
        return tf.squeeze(action, axis=-1)


class ActorCritic(tf.keras.Model):
    def __init__(self, action_shape, Encoder):
        super(ActorCritic, self).__init__()
        self.cell = ActorCriticCell(action_shape, Encoder)
        self.rnn = tf.keras.layers.RNN(self.cell, return_sequences=True)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return self.rnn(inputs, mask=mask, training=training, initial_state=initial_state)