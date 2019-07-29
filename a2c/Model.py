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


class ActorCritic(tf.keras.Model):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.conv = kl.Conv2D(32, 3, activation='relu')
        self.flatten = kl.Flatten()
        self.d1 = kl.Dense(128, activation='relu')
        self.d2 = kl.Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv(x)
        x = self.flatten(x)

        action_param = self.d1(x)
        values = self.d2(x)
        return action_param, values

    def action_value(self, state):
        logits, value = self.predict(state)
        action = self.dist.predict(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)
