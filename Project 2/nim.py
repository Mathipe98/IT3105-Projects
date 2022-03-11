from typing import Tuple
import numpy as np

class Nim:

    def __init__(self, n: int=10, k: int=3) -> None:
        self.start_n = n
        self.start_k = k
        self.n = n
        self.k = k
        self.current_state = None

    def reset(self) -> np.ndarray:
        self.n = self.start_n
        self.k = self.start_k
        self.current_state = np.array([self.n])
        return self.current_state
    
    def get_legal_actions(self, state: np.ndarray) -> np.ndarray:
        return np.arange(1, min(self.k+1, state[0]+1))
    
    def step(self, action: int) -> Tuple[np.ndarray, int, bool]:
        if action > self.n:
            raise ValueError(f"Cannot remove {action} items from a pile of size {self.n}")
        self.n = self.n - action
        next_state = np.array([self.n])
        if self.is_winning(next_state):
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        self.current_state = next_state
        return self.current_state, reward, done
    
    def is_winning(self, state: np.ndarray) -> bool:
        return state[0] == 0
    
    # def is_losing(self, state: np.ndarray) -> bool:
    #     return state[0] == 0

    
if __name__ == "__main__":
    test = Nim()
    state = test.reset()
    done = test.is_winning(state) or test.is_losing(state)
    while not done:
        print(state)
        actions = test.get_legal_actions(state)
        action = actions[0]
        print(action)
        state, reward, done = test.step(action)