from typing import List
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.layers import Dense, InputLayer

np.random.seed(123)
tf.random.set_seed(123)

def create_model(n_input: int, hidden_layers: tuple, hl_activations: List, n_output: int, output_activation: str, optimizer: str, lr: float) -> tf.keras.Sequential:
    assert len(hidden_layers) == len(hl_activations), "An activation function must be provided for each hidden layer"
    model = keras.Sequential()
    model.add(InputLayer(input_shape=(n_input,)))
    for dim, activation in zip(hidden_layers, hl_activations):
        model.add(Dense(dim, activation=activation))
    model.add(Dense(n_output, activation=output_activation))
    if optimizer == 'Adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
    elif optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.5)
    elif optimizer == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, momentum=0.5)
    elif optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        raise ValueError(f'Invalid optimizer argument: {optimizer}')
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    input_shape = (50,)
    hidden_layers = (10,20,5,2)
    hl_activations = ('linear', 'sigmoid', 'tanh', 'relu')
    output_layer = 20
    output_activation = 'softmax'
    optimizer = 'Adam'
    lr = 0.01
    test_model = create_model(input_shape, hidden_layers, hl_activations, output_layer, output_activation, optimizer, lr)
    test_model.summary()
