from typing import Any
import numpy as np
import random


class Actor:

    def __init__(self, alpha: float, gamma: float, lamb: float,
                epsilon: float, game: Any) -> None:
        self.alpha, self.gamma, self.lamb = alpha, gamma, lamb
        self.epsilon = epsilon
        self.PI = {}
        self.eligibility_trace = {}
        # self.PI = np.zeros((state_buckets + (self.n_actions,)))
        # self.eligibility_trace = np.zeros(shape=(self.n_states, self.n_actions))
        # self.eligibility_trace = np.zeros(state_buckets + (self.n_actions,))
        self.debug = 0
        self.game = game
        self.initialize()
    
    def initialize(self) -> None:
        states = self.game.states
        for state in states:
            actions = self.game.get_legal_actions(state)
            for action in actions:
                SAP = (state, action)
                self.PI[SAP] = 0
                self.eligibility_trace[SAP] = 0
        pass

    def update(self, state: int, action: int, delta: float) -> None:
        """Updating PI(s, a)

        Args:
            delta (float): [description]
            state (int): [description]
            action (int): [description]
        """
        prediction = self.PI[(state,action)]
        adjustment = self.alpha * delta * self.eligibility_trace[(state,action)]
        self.PI[(state,action)] = prediction + adjustment

    def set_eligibility(self, state: int, action: int=None) -> None:
        self.eligibility_trace[(state,action)] = 1
    
    def update_eligibility(self, state: int, action: int=None) -> None:
        self.eligibility_trace[(state,action)] = self.gamma * self.lamb * self.eligibility_trace[(state,action)]

    def reset_eligibilities(self) -> None:
        states = self.game.states
        for state in states:
            actions = self.game.get_legal_actions(state)
            for action in actions:
                SAP = (state, action)
                self.eligibility_trace[SAP] = 0
    
    def get_action(self, state) -> int:
        actions = self.game.get_legal_actions(state)
        if np.random.uniform(0,1) < self.epsilon:
             return random.choice(actions)
        possible_actions = {(state, action): self.PI[(state, action)] for action in actions}
        return max(possible_actions, key=possible_actions.get)[1]

    def calculate_delta(self, r: float, s1: int, s2: int, a1: int, a2: int) -> float:
        Q1 = self.PI[s1][a1]
        Q2 = self.PI[s2][a2]
        delta = r + (self.gamma * Q2) - Q1
        return delta

def test_stuff():
    actor = Actor()
    actor.initialize()
    action = actor.calculate_best_action((1,1,2,2))
    print(action)

if __name__ == "__main__":
    test_stuff()
