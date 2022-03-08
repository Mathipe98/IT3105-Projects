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
    
    def step(self, action: int) -> np.ndarray:
        if action > self.n:
            raise ValueError(f"Cannot remove {action} items from a pile of size {self.n}")
        return np.array([self.n - action])
    
    def is_winning(self, state: np.ndarray) -> bool:
        return state[0] <= self.k

    
if __name__ == "__main__":
    test = Nim()
    state = test.reset()
    print(state)
    actions = test.get_legal_actions(state)
    print(actions)
    a1 = actions[2]
    new_state = test.step(a1)
    print(new_state)