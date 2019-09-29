"""
File: tensorboard
Date: 5/6/19 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import tensorflow as tf

class Tensorboard:

    def __init__(self, logdir):
        if logdir is not None:
            self.set_directory(logdir)

    def set_directory(self, logdir):
        self.writer = tf.summary.FileWriter(logdir)

    def log_scalar(self, tag, value, global_step):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.writer.add_summary(summary, global_step=global_step)

    def close(self):
        self.writer.close()

    def flush(self):
        self.writer.flush()

