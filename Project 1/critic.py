from typing import Any, List
import numpy as np
import tensorflow as tf


class Critic:

    def __init__(self, alpha: float, gamma: float, lamb: float,
                 game: Any, use_nn: bool = False, network_dim: tuple = None) -> None:
        self.alpha, self.gamma, self.lamb = alpha, gamma, lamb
        self.game = game
        # Neural network stuff
        self.use_nn = use_nn
        self.network_dim = network_dim
        self.model = None
        self.id_matrix = None
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
        states = self.game.states
        if self.use_nn:
            self.id_matrix = np.identity(self.game.n_states)
            model = tf.keras.Sequential()
            for i in range(len(self.network_dim) - 1):
                dimension = self.network_dim[i]
                model.add(tf.keras.layers.Dense(dimension), activation='relu')
            model.add(tf.keras.layers.Dense(
                self.network_dim[-1], activation='linear'))
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha), loss='mse', metrics=['mae'])
            self.model = model
        else:
            for state in states:
                actions = self.game.get_legal_actions(state)
                for action in actions:
                    SAP = (state, action)
                    self.values[SAP] = np.random.uniform(low=-0.1, high=0.1)
                    self.eligibility_trace[SAP] = 0

    def calculate_delta(self, r: float, s1: int, s2: int, a1: int = None, a2: int = None) -> float:
        if not self.use_nn:
            V1 = self.values[(s1, a1)]
            V2 = self.values[(s2, a2)]
            delta = r + (self.gamma * V2) - V1
        else:
            s1_one_hot = self.id_matrix[s1:s1+1]
            s2_one_hot = self.id_matrix[s2:s2+1]
            V1 = self.model.predict(s1_one_hot)
            V2 = self.model.predict(s2_one_hot)
            delta = r + self.gamma *V2 - V1
            delta = delta[0][0]
        return delta

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
    
    

    def step_update_nn(self, s1: int, s2: int) -> None:
        n_states = self.id_matrix.shape[0]
        s1_hot = self.id_matrix[s1:s1+1].reshape(n_states,)
        s2_hot = self.id_matrix[s2:s2+1].reshape(n_states,)
        future_reward = self.model.predict(s2_hot)


    def batch_update_nn(self, visited: List) -> None:
        n_states = self.id_matrix.shape[0]
        current_states = np.array([v[0] for v in visited])
        inputs = np.array([self.id_matrix[s:s+1].reshape(n_states,)
                          for s in current_states])

        next_states = np.array([v[2] for v in visited])
        next_inputs = np.array(
            [self.id_matrix[s:s+1].reshape(n_states,) for s in next_states])

        predictions = self.model(next_inputs)
        rewards = np.array([v[4] for v in visited]).reshape(predictions.shape)
        discounted = self.gamma * predictions
        targets = np.add(rewards, discounted)
        self.model.fit(inputs, targets,
                       batch_size=inputs.shape[0], epochs=1, verbose=0)

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


def test_stuff():
    critic = Critic()
    critic.initialize()
    print()


if __name__ == "__main__":
    test_stuff()
