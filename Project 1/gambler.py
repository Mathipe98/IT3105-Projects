from typing import List, Tuple
import numpy as np
import random
# import tensorflow as tf
from collections import deque
from itertools import product, permutations


class Gambler:

    def __init__(self, p_win: float=0.5, goal_cash: int=100) -> None:
        self.p_win = p_win
        self.goal_cash = goal_cash
        self.states = None
        self.current_state = None
        self.current_state_parameters = None
        self.initialize()
    
    def initialize(self) -> None:
        self.generate_all_states()
        self.reset()

    def reset(self) -> None:
        self.current_state = np.random.randint(1, self.goal_cash-1)
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, int, int]:
        if np.random.uniform(0,1) < self.p_win:
            next_state = min(self.goal_cash, self.current_state + action)
        else:
            next_state = max(0, self.current_state - action)
        if next_state == self.goal_cash:
            reward = 1
            done = True
        else:
            reward = 0
            done = next_state == 0
        self.current_state = next_state
        return next_state, reward, done

    def encode_state(self, state: int) -> np.ndarray:
        pass

    def decode_state(self, encoded_state: np.ndarray) -> int:
        pass

    def get_legal_actions(self, state: int) -> List:
        # Workaround for calculating next_action in training; when we have 0 or max money, there is no next
        # action. But the algorithm expects it. Therefore just return an arbitrary value that won't be used
        # in the update
        if state == 0 or state == 100:
            return np.array([0])
        end = min(state, self.goal_cash - state) + 1
        return np.array([cash_spent for cash_spent in range(1, end)])

    def is_winning(self, state: int) -> bool:
        return state == self.goal_cash

    def generate_all_states(self) -> None:
        self.states = np.array([i for i in range(0, self.goal_cash + 1)])



def test_stuff():
    gambler_params = {
        "p_win": 1.0,
        "goal_cash": 100
    }
    gambler = Gambler(**gambler_params)
    print(f"Gambler starting cash: {gambler.current_state}")
    actions = gambler.get_legal_actions(gambler.current_state)
    random_action = random.choice(actions)
    print(f"Gambler chose to bet: {random_action}")
    gambler.step(random_action)
    print(f"Money after bet: {gambler.current_state}")

if __name__ == "__main__":
    test_stuff()