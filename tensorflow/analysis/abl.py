import tensorflow as tf
import numpy as np


def min_max_normalization(x):
    x_min = tf.reduce_min(x)
    x_max = tf.reduce_max(x)
    norm = (x - x_min) / (x_max - x_min)
    return norm


class ABLAnalysis():
    def __init__(self):
        return

    def analysis(self, data):
        """
            data np.array
            sample-wise training loss shape (epoch, n_samples)
            return first 20 epoch avgs for each sample
        """
        # lower for bd, use 1 - score
        return 1 - min_max_normalization(tf.constant(data[:20, :].mean(axis=0))).numpy()