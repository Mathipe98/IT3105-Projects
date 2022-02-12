import numpy as np


class Actor:

    def __init__(self, alpha: float, gamma: float, lamb: float,
                epsilon: float, n_states: int, n_actions: int) -> None:
        self.alpha, self.gamma, self.lamb = alpha, gamma, lamb
        self.epsilon = epsilon
        self.n_states, self.n_actions = n_states, n_actions
        # Let policy = PI(s), while PI = PI(s,a)
        # policy = 1D array; PI = 2D array
        self.policy = None
        self.PI = None
        self.eligibility_trace = None
        self.debug = 0
        self.initialize()

    def initialize(self) -> None:
        self.PI = np.zeros((self.n_states, self.n_actions))
        self.eligibility_trace = np.zeros(shape=(self.n_states, self.n_actions))
    
    def update(self, delta: float, state: int, action: int) -> None:
        """Updating PI(s, a)

        Args:
            delta (float): [description]
            state (int): [description]
            action (int): [description]
        """
        # update_val = self.PI[state, action] + self.alpha * delta
        # old_val = self.PI[state, action]
        # self.PI[state, action] = update_val
        # new_val = self.PI[state, action]
        # print()
        self.PI[state, action] = self.PI[state, action] + self.alpha * delta# * self.eligibility_trace[state, action]
        # self.update_policy_map()
    
    def set_eligibility(self, state: int, action: int=None) -> None:
        self.eligibility_trace[state, action] = 1
    
    def update_eligibility(self, state: int, action: int=None) -> None:
        self.eligibility_trace[state, action] = self.gamma * self.lamb * self.eligibility_trace[state, action]

    def reset_eligibilities(self) -> None:
        self.eligibility_trace = np.zeros(shape=(self.n_states, self.n_actions))
    
    def get_action(self, state) -> int:
        if np.random.uniform(0,1) < self.epsilon:
            return np.random.choice(range(self.n_actions))
        return np.argmax(self.PI[state, :])

    def calculate_delta(self, r: float, s1: int, s2: int, a1: int, a2: int) -> float:
        delta = r + self.gamma * self.PI[s2,a2] - self.PI[s1, a1]
        return delta

def test_stuff():
    actor = Actor()
    actor.initialize()
    action = actor.calculate_best_action(50)
    for i in range(100):
        print(actor.calculate_best_action(40))

if __name__ == "__main__":
    test_stuff()
