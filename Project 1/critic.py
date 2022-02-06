import numpy as np
from actor import Actor


class Critic:

    def __init__(self, alpha: float, gamma: float,
                 n_states: int, n_actions: int, use_nn: bool = False) -> None:
        self.alpha, self.gamma = alpha, gamma
        self.n_states, self.n_actions = n_states, n_actions
        self.use_nn = use_nn
        # If we're using neural network, then state eval regards only state
        # If we're using SAP, then the value must be tied to both state and action
        self.values = None

    def initialize(self) -> None:
        if self.use_nn:
            self.values = np.random.uniform(
                low=-0.1, high=0.1, size=(self.n_states,))
        else:
            self.values = np.random.uniform(
                low=-0.1, high=0.1, size=(self.n_states, self.n_actions))

    def calculate_delta(self, r: float, s1: int, s2: int, a1: int = None, a2: int = None) -> float:
        if not self.use_nn:
            v_s = self.values[s1, a1]
            v_s_prime = self.values[s2, a2]
            delta = r + self.gamma * v_s_prime - v_s
            return delta
        pass


def test_stuff():
    critic = Critic()
    critic.initialize()
    print()


if __name__ == "__main__":
    test_stuff()
