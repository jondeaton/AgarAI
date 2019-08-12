"""
File: Actor
Date: 2019-07-28 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import tensorflow as tf
import tensorflow.keras.layers as kl

import numpy as np


class CNNEncoder(tf.keras.Model):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = kl.Conv2D(8, 3, 1, activation=tf.nn.leaky_relu)
        self.conv2 = kl.Conv2D(8, 3, 1, activation=tf.nn.leaky_relu)
        self.conv3 = kl.Conv2D(8, 3, 1, activation=tf.nn.leaky_relu)
        self.flatten = kl.Flatten()
        self.dropout = kl.Dropout(0.07)

    def call(self, x):
        x = tf.dtypes.cast(x, tf.float32)
        x = self.dropout(self.conv1(x))
        x = self.dropout(self.conv2(x))
        x = self.dropout(self.conv3(x))
        return self.flatten(x)


class DenseEncoder(tf.keras.Model):
    def __init__(self):
        super(DenseEncoder, self).__init__()
        self.dense1 = kl.Dense(16, activation=tf.nn.relu)
        self.dense2 = kl.Dense(16, activation=tf.nn.relu)
        self.dense3 = kl.Dense(16, activation=tf.nn.relu)
        self.dropout = kl.Dropout(0.1)

    def call(self, x):
        x = self.dense1(x)
        x = self.dropout(self.dense2(x))
        x = self.dropout(self.dense3(x))
        return x


class ActorCritic(tf.keras.Model):
    def __init__(self, action_shape, Encoder):
        super(ActorCritic, self).__init__()

        self.encoder = Encoder()
        self.dense = kl.Dense(128, activation=tf.nn.leaky_relu)
        self.dropout = kl.Dropout(0.05)

        self.value_layer = kl.Dense(1, activation=None, name="value")
        self.action_layer = kl.Dense(np.prod(action_shape), activation=None, name="action")

    def call(self, x: tf.Tensor):
        x = self.encoder(x)
        x = self.dropout(self.dense(x))

        action_param = self.action_layer(x)
        values = self.value_layer(x)
        return action_param, values

    def action_value(self, state):
        logits, value = self.predict(state)
        action = tf.random.categorical(logits, 1)
        return tf.squeeze(action, axis=-1), np.squeeze(value, axis=1)
