"""
File: Actor
Date: 2019-07-28 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import tensorflow as tf
import tensorflow.keras.layers as kl

import numpy as np


class Distribution(tf.keras.Model):
    def call(self, logits):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class CNNEncoder(tf.keras.Model):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = kl.Conv2D(2, 8, 4, activation=tf.nn.relu)
        self.conv2 = kl.Conv2D(2, 3, 1, activation=tf.nn.relu)
        self.flatten = kl.Flatten()

    def call(self, x):
        x = tf.dtypes.cast(x, tf.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.flatten(x)


class DenseEncoder(tf.keras.Model):
    def __init__(self):
        super(DenseEncoder, self).__init__()
        self.dense1 = kl.Dense(4, activation=tf.nn.relu)
        # self.dense2 = kl.Dense(16, activation=tf.nn.relu)
        # self.dense3 = kl.Dense(16, activation=tf.nn.relu)
        # self.dropout = kl.Dropout(0.1)

    def call(self, x):
        x = self.dense1(x)
        # x = self.dropout(self.dense2(x))
        # x = self.dropout(self.dense3(x))
        return x


class ActorCritic(tf.keras.Model):
    def __init__(self, action_shape, Encoder):
        super(ActorCritic, self).__init__()

        self.encoder = Encoder()
        self.dense = kl.Dense(16, activation=tf.nn.relu)

        self.dropout = kl.Dropout(0.1)

        self.value_layer = kl.Dense(1, activation=None, name="value")

        self.action_shape = action_shape
        action_size = np.prod(action_shape)
        self.action_layer = kl.Dense(action_size, activation=None,
                                     name="action")

        self.action_distribution = Distribution()

    def call(self, x: tf.Tensor):
        x = self.encoder(x)
        x = self.dense(x)
        x = self.dropout(x)

        action_param = self.action_layer(x)
        values = self.value_layer(x)
        return action_param, values

    def action_value(self, state):
        logits, value = self.predict(state)
        action = self.action_distribution.predict(logits)
        return action, np.squeeze(value, axis=1)
