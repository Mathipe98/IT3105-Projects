import numpy as np
from actor import Actor


class Critic:

    def __init__(self, alpha: float, gamma: float, lamb: float,
                 n_states: int, n_actions: int, use_nn: bool = False) -> None:
        self.alpha, self.gamma, self.lamb = alpha, gamma, lamb
        self.n_states, self.n_actions = n_states, n_actions
        self.use_nn = use_nn
        # If we're using neural network, then state eval regards only state
        # If we're using SAP, then the value must be tied to both state and action
        self.values = None
        self.eligibility_trace = None
        self.initialize()

    def initialize(self) -> None:
        """Method for initializing the critic with a value function.
        If we're using a neural network (function approximator), then we use a
        1-dimensional value function (only based on state).
        If we're using table-lookup, then we create a 2D array indexed by the state
        AND the action performed in that state.
        """
        if self.use_nn:
            self.values = np.random.uniform(
                low=-0.1, high=0.1, size=(self.n_states,))
            self.eligibility_trace = np.zeros(self.n_states)
        else:
            self.values = np.random.uniform(
                low=-0.1, high=0.1, size=(self.n_states, self.n_actions))
            # self.values = np.zeros(shape=(self.n_states, self.n_actions))
            self.eligibility_trace = np.zeros(shape=(self.n_states, self.n_actions))

    def calculate_delta(self, r: float, s1: int, s2: int, a1: int = None, a2: int = None) -> float:
        if not self.use_nn:
            delta = r + self.gamma * self.values[s2, a2] - self.values[s1, a1]
            return delta
        pass

    def update(self, delta: float, state: int, action: int=None) -> None:
        """Method for updating the value function of the critic by the delta value

        Args:
            delta (float): Calculated discrepancy between result and prediction
            state (int): Current state for which to update the value function
            action (int): The action performed in the current state, only used if use_nn is False.
                        Defaults to None.
        """
        if not self.use_nn:
            self.values[state,action] = self.values[state,action] + self.alpha * delta * self.eligibility_trace[state, action]
        pass

    def set_eligibility(self, state: int, action: int=None) -> None:
        if not self.use_nn:
            self.eligibility_trace[state, action] = 1
    
    def update_eligibility(self, state: int, action: int=None) -> None:
        if not self.use_nn:
            self.eligibility_trace[state, action] = self.gamma * self.lamb * self.eligibility_trace[state, action]
    
    def reset_eligibilities(self) -> None:
        self.eligibility_trace = np.zeros(shape=(self.n_states, self.n_actions))


def test_stuff():
    critic = Critic()
    critic.initialize()
    print()


if __name__ == "__main__":
    test_stuff()
