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
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    elif optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        raise ValueError(f'Invalid optimizer argument: {optimizer}')
    model.compile(loss='kl_divergence', optimizer=optimizer, metrics=['accuracy'])
    return model

class LiteModel:
    
    @classmethod
    def from_file(cls, model_path):
        return LiteModel(tf.lite.Interpreter(model_path=model_path))
    
    @classmethod
    def from_keras_model(cls, kmodel):
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))
    
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det["index"]
        self.output_index = output_det["index"]
        self.input_shape = input_det["shape"]
        self.output_shape = output_det["shape"]
        self.input_dtype = input_det["dtype"]
        self.output_dtype = output_det["dtype"]
        
    def predict(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i:i+1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.output_index)[0]
        return out
    
    def predict_single(self, inp):
        """ Like predict(), but only for a single record. The input data can be a Python list. """
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out[0]

if __name__ == '__main__':
    input_shape = 50
    hidden_layers = (10,20,5,2)
    hl_activations = ('linear', 'sigmoid', 'tanh', 'relu')
    output_layer = 20
    output_activation = 'softmax'
    optimizer = 'Adam'
    lr = 0.01
    test_model = create_model(input_shape, hidden_layers, hl_activations, output_layer, output_activation, optimizer, lr)
    test_model.summary()