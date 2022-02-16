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
        """Constructor for the Critic in the Actor-Critic algorithm.

        Args:
            alpha (float): Learning rate of the critic
            gamma (float): Decay rate of future rewards
            lamb (float): Decay rate of the eligibility trace
            game (Any): Object containing the game to play. Only uses generic method/function calls.
            use_nn (bool, optional): Whether to use a neural network. Defaults to False.
            network_dim (tuple, optional): Dimension of potential neural network. Defaults to None.
        """
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
            self.model = create_model(
                self.game.enc_shape, self.network_dim, self.alpha)
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
        """Method for setting the eligibility value of 1 to the
        current state-action pair. Here we do not use accumulative
        eligibility traces, but rather the max variant

        Args:
            SAP (tuple): State-action pair
        """
        self.eligibility_trace[SAP] = 1

    def update_eligibility(self, SAP: tuple) -> None:
        """Updating the eligibility trace of the current 
        state-action pair in a decaying manner

        Args:
            SAP (tuple): State-action pair
        """
        self.eligibility_trace[SAP] = self.gamma * \
            self.lamb * self.eligibility_trace[SAP]

    def reset_eligibilities(self) -> None:
        """Method for resetting all eligibility traces
        when a new episode starts.
        """
        states = self.game.states
        for state in states:
            actions = self.game.get_legal_actions(state)
            for action in actions:
                SAP = (state, action)
                self.eligibility_trace[SAP] = 0

    def evaluate_state(self, state: np.ndarray) -> float:
        """Generic method that takes in a state, and passes it
        into a neural network such that the network can evaluate
        the value of the state.

        Args:
            state (np.ndarray): (Encoding of) the state in question

        Returns:
            float: The predicted value of the state
        """
        prediction = self.model(state)
        assert prediction.shape == (1,) or prediction.shape == (
            1, 1), "Prediction shapes are weird"
        return tf.get_static_value(prediction[0][0])

    def train(self, replay_memory: deque) -> None:
        """Method for training the neural network with previously observed
        data. We use a batch size of 128 where we randomly select 128
        state and target pairs, and feed this into the neural network to train
        on.

        Args:
            replay_memory (deque): List (queue) that contains a large number of previous examples.
                                    We randomly select 128 items from this large memory
        """
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
        self.model.fit(inputs, valid, batch_size=batch_size,
                       verbose=0, shuffle=True)
