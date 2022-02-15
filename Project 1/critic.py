from collections import deque
import random
from typing import Any, List
import numpy as np
import tensorflow as tf
from network import create_model

tf.random.set_seed(123)


class Critic:

    def __init__(self, alpha: float, gamma: float, lamb: float,
                 game: Any, use_nn: bool = False, network_dim: tuple = None) -> None:
        self.alpha, self.gamma, self.lamb = alpha, gamma, lamb
        self.game = game
        # Neural network stuff
        self.use_nn = use_nn
        self.network_dim = network_dim
        self.model = None
        self.episode_targets = []
        # If we're using neural network, then state eval regards only state
        # If we're using SAP, then the value must be tied to both state and action
        self.values = {}
        self.eligibility_trace = {}
        self.initialize()

    def initialize(self) -> None:
        """Method for initializing the critic with a value function.
        If we're using a neural network (function approximator), then we use a
        1-dimensional value function (only based on state).
        If we're using table-lookup, then we create a 2D array indexed by the state
        AND the action performed in that state.
        """
        if self.use_nn:
            self.model = create_model(self.game.enc_shape, self.network_dim, self.alpha)
        else:
            states = self.game.states
            for state in states:
                actions = self.game.get_legal_actions(state)
                for action in actions:
                    SAP = (state, action)
                    self.values[SAP] = np.random.uniform(low=-0.1, high=0.1)
                    self.eligibility_trace[SAP] = 0

    def table_update(self, SAP: tuple, delta: float) -> None:
        """Method for updating the value function of the critic by the delta value

        Args:
            delta (float): Calculated discrepancy between result and prediction
            state (int): Current state for which to update the value function
            action (int): The action performed in the current state, only used if use_nn is False.
                        Defaults to None.
        """
        prediction = self.values[SAP]
        adjustment = self.alpha * delta * self.eligibility_trace[SAP]
        self.values[SAP] = prediction + adjustment

    def set_eligibility(self, SAP: tuple) -> None:
        self.eligibility_trace[SAP] = 1

    def update_eligibility(self, SAP: tuple) -> None:
        self.eligibility_trace[SAP] = self.gamma * \
            self.lamb * self.eligibility_trace[SAP]

    def reset_eligibilities(self) -> None:
        states = self.game.states
        for state in states:
            actions = self.game.get_legal_actions(state)
            for action in actions:
                SAP = (state, action)
                self.eligibility_trace[SAP] = 0
    
    def evaluate_state(self, state: np.ndarray) -> float:
        prediction = self.model(state)
        assert prediction.shape == (1,) or prediction.shape == (1,1), "Prediction shapes wonky"
        return tf.get_static_value(prediction[0][0])
    
    def train(self, replay_memory: deque) -> None:
        batch_size = 128
        mini_batch = random.sample(replay_memory, batch_size)
        inputs = []
        valid = []
        for encoded_state, target, reward, done in mini_batch:
            inputs.append(encoded_state)
            # If the state lead to a final state, then we don't propagate the future reward (target), only the immediate reward
            if done:
                valid.append(reward)
            else:
                valid.append(target)
        inputs = np.array(inputs)
        valid = np.array(valid)
        self.model.fit(inputs, valid, batch_size=batch_size, verbose=0, shuffle=True)
            
    


def test_stuff():
    critic = Critic()
    critic.initialize()
    print()


if __name__ == "__main__":
    test_stuff()
