import numpy as np
import copy
from typing import Tuple
from node import Node

class Nim:

    def __init__(self, n: int=10, k: int=3) -> None:
        self.n = n
        self.k = k
        self.current_state = None
        self.current_node = None

    def reset(self) -> Node:
        self.k = self.k
        self.current_state = np.array([self.n])
        self.current_node = Node(game=self, state=self.current_state)
        self.current_node.visits = 1
        return self.current_node
    
    def get_n_possible_actions(self) -> int:
        return self.k
    
    def get_legal_actions(self, node: Node) -> np.ndarray:
        state = node.state
        return np.arange(1, min(self.k+1, state[0]+1))
    
    def step(self, action: int) -> Tuple[Node, int, bool]:
        if action > self.current_state[0]:
            raise ValueError(f"Cannot remove {action} items from a pile of size {self.n}")
        next_state = self.current_state - action
        self.current_state = next_state
        next_node = Node(game=self, state=next_state, parent=self.current_node, parent_action=action)
        if self.is_winning(next_node):
            reward = 10
            done = True
        elif self.is_losing(next_node):
            reward = -5
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
        return node.state[0] <= self.k
    
    def is_losing(self, node: Node) -> bool:
        return node.state[0] == 0
    
    def evaluate_state(self, node: Node) -> int:
        if self.is_winning(node):
            return 10
        elif self.is_losing(node):
            return -5
        return 0
    
    def encode_state(self, node: Node, player: int) -> np.ndarray:
        """Method to one-hot encode the state of the Nim-game.
        This includes the player number.

        Args:
            node (Node): Node corresponding to current state
            player (int): Integer describing either first or second player

        Returns:
            np.ndarray: One-hot encoding of the state + player
        """
        state = node.state
        encoded_state = np.zeros(self.n + 1)
        index = state[0] - 1
        encoded_state[index] = 1
        encoded_state[-1] = player
        return encoded_state

    
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