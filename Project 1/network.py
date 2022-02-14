import tensorflow as tf
import numpy as np
from tensorflow import keras

from collections import deque
import time
import random


def model(state_shape: tuple, dimensions: tuple, lr: float) -> tf.keras.Sequential:
    """ The agent maps X-states to Y-output
    e.g. The neural network output is [0.4], which is the
    predicted value of state X.
    """
    # Initialize the network weights
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    for i in range(len(dimensions)):
        dimension = dimensions[i]
        if i == 0:
            model.add(keras.layers.Dense(dimension, input_shape=state_shape, activation='relu', kernel_initializer=init))
        elif i == len(dimensions) - 1:
            model.add(keras.layers.Dense(dimension, activation='linear', kernel_initializer=init))
        else:
            model.add(keras.layers.Dense(dimension, activation='relu', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])
    return model


if __name__ == '__main__':
    dims = (5,1)
    lr = 0.001
    state_shape = (5,)
    test = model(state_shape, dims, lr)