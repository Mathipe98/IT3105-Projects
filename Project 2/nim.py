import numpy as np
import copy
from typing import Tuple
from node import Node

class Nim:

    def __init__(self, n: int=10, k: int=3) -> None:
        self.n = n
        self.k = k
        self.current_state = np.array([self.n])
        self.current_node = Node(game=0, state=self.current_state)
        self.current_node.visits = 1

    def reset(self) -> Node:
        self.k = self.k
        self.current_state = np.array([self.n])
        self.current_node = Node(game=0, state=self.current_state)
        self.current_node.visits = 1
        return self.current_node
    
    def get_legal_actions(self, node: Node) -> np.ndarray:
        state = node.state
        return np.arange(1, min(self.k+1, state[0]+1))
    
    def step(self, action: int) -> Tuple[Node, int, bool]:
        if action > self.current_state[0]:
            raise ValueError(f"Cannot remove {action} items from a pile of size {self.n}")
        next_state = self.current_state - action
        self.current_state = next_state
        next_node = Node(game=0, state=next_state, parent=self.current_node, parent_action=action)
        if self.is_winning(next_node):
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        self.current_node = next_node
        return self.current_node, reward, done
    
    def simulate_action(self, action: int, node: Node) -> Tuple[Node, int, bool]:
        self_copy = copy.copy(self)
        self_copy.current_node = node
        self_copy.current_state = node.state
        return self_copy.step(action)
    
    def is_winning(self, node: Node) -> bool:
        return node.state[0] == 0
    
    # def is_losing(self, state: np.ndarray) -> bool:
    #     return state[0] == 0

    
if __name__ == "__main__":
    test = Nim()
    print(test.current_state)
    node = test.reset()
    actions = test.get_legal_actions(node)
    print(actions)
    done = False
    while not done:
        action = np.random.choice(actions)
        node, reward, done = test.step(action)
        print(f"Action taken: {action}")
        print(node.state)
        actions = test.get_legal_actions(node)
    node = test.reset()
    simulated_node, reward, done = test.simulate_action(2, node)
    print(f"Simulated node state: {simulated_node.state}")
    print(f"Actual node state: {node.state}")