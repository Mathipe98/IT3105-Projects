import numpy as np


class Actor:

    def __init__(self, epsilon: float, alpha: float, gamma: float,
                n_states: int, n_actions: int) -> None:
        self.epsilon, self.alpha, self.gamma = epsilon, alpha, gamma
        self.n_states, self.n_actions = n_states, n_actions
        # Let policy = PI(s), while PI = PI(s,a)
        # policy = 1D array; PI = 2D array
        self.policy = None
        self.PI = None
        self.debug = 0

    def initialize(self) -> None:
        self.PI = np.zeros((self.n_states, self.n_actions),dtype=int)
        self.policy = np.zeros(self.n_states,dtype=int)
        self.update_policy()
    
    def update_policy(self) -> None:
        update = lambda state: self.calculate_best_action(state)
        self.policy = np.array(list(map(update, self.policy)))
    
    def calculate_best_action(self, state: int) -> int:
        # With probability epsilon, choose a random action
        if np.random.uniform(0,1) < self.epsilon:
            return np.random.choice(range(self.n_actions))
        actions = self.PI[state]
        return np.argmax(actions)
    
    def get_best_action(self, state) -> int:
        return self.policy[state]

    # def perform_action(self, action: int) -> None:
    #     print(f"State before taking an action:\n\t {self.game.current_state}")
    #     next_state, reward = self.game.get_next_state(action)
    #     print(f"State after taking an action:\n\t {self.game.current_state}")
        


def test_stuff():
    actor = Actor()
    actor.initialize()
    action = actor.calculate_best_action(50)
    for i in range(100):
        print(actor.calculate_best_action(40))

if __name__ == "__main__":
    test_stuff()
