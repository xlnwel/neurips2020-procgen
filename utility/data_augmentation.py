import random
import numpy as np
import tensorflow as tf

tfi = tf.image
tki = tf.keras.preprocessing.image

def pad_crop(x, pad, mode, prob):
    if random.random() < prob:
        h, w, c = x.shape[1:]
        paddings = [[0, 0], [pad, pad], [pad, pad], [0, 0]]
        if isinstance(x, np.ndarray):
            x = np.pad(x, paddings, mode=mode)
            sh = np.random.randint(0, 2*pad)
            sw = np.random.randint(0, 2*pad)
            x = x[:, sh:sh+h, sw:sw+w, :]
        elif isinstance(x. tf.Tensor):
            paddings = tf.constant(paddings)
            x = tf.pad(x, paddings)
            x = tfi.random_crop(x, (-1, h, w, c))
        else:
            raise ValueError(x)
    return x
