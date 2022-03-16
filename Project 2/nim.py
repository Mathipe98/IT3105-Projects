import numpy as np
import copy
from typing import Tuple

class Nim:

    def __init__(self, n: int=10, k: int=3) -> None:
        self.n = n
        self.k = k
        self.current_state = np.array([self.n])

    def reset(self) -> np.ndarray:
        self.k = self.k
        self.current_state = np.array([self.n])
        return self.current_state
    
    def get_legal_actions(self, state: np.ndarray) -> np.ndarray:
        return np.arange(1, min(self.k+1, state[0]+1))
    
    def step(self, action: int) -> Tuple[np.ndarray, int, bool]:
        if action > self.current_state[0]:
            raise ValueError(f"Cannot remove {action} items from a pile of size {self.n}")
        next_state = self.current_state - action
        if self.is_winning(next_state):
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        self.current_state = next_state
        return self.current_state, reward, done
    
    def simulate_action(self, action: int, state: np.ndarray) -> np.ndarray:
        self_copy = copy.copy(self)
        self_copy.current_state = state
        return self_copy.step(action)
    
    def is_winning(self, state: np.ndarray) -> bool:
        return state[0] == 0
    
    # def is_losing(self, state: np.ndarray) -> bool:
    #     return state[0] == 0

    
if __name__ == "__main__":
    test = Nim()
    state = test.reset()
    print(state)
    done = test.is_winning(state)
    while not done:
        actions = test.get_legal_actions(state)
        action = actions[0]
        print(action)
        state, reward, done = test.step(action)
        print(state)