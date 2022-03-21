import numpy as np
import tensorflow as tf
import copy
from tensorflow import keras
from keras import Sequential
from collections import deque
from node import Node
from nim import Nim

np.random.seed(123)
class MCTSAgent:

    def __init__(self, game: Nim, M: int, model: Sequential, player: int=1, epsilon: float=0.1) -> None:
        self.game = game
        self.M = M
        self.tree = {}
        self.leaf_nodes = []
        self.multiplier = 1 if player == 1 else -1
    
    def train(self):
        replay_buffer = deque(maxlen=100000)
        while True:
            current_state = self.game.reset()
        
    def is_leaf_node(self, current_node: Node, new_state: np.ndarray=None) -> bool:
        if len(current_node.children) == 0:
            return True
        # child_node = next(child for child in current_node.children if child.state == new_state)
        # if child_node is None:
        #     return True
        return False

    def rollout(self, current_node: Node) -> None:
        first_node = current_node
        trajectory = [current_node]
        reward_multiplier = 1
        done = False
        while not done:
            action_distribution = self.model(current_node.state).numpy()
            # Do something with the legal actions to cut out the illegal actions that the network returns
            legal_actions = self.game.get_legal_actions(current_node.state)
            action = np.argmax(action_distribution)
            current_node, reward, done = self.game.simulate_action(action, current_node)
            trajectory.append(current_node)
            reward_multiplier *= -1
            if done:
                self.backprop(first_node, reward, reward_multiplier)

    def backprop(self, node: Node, value: int, multiplier: int) -> None:
        while True:
            node.value += value * multiplier
            multiplier *= -1
            node = node.parent
            if node is None:
                break
        
    def tree_traversal(self, root_node: Node) -> None:
        current_node = root_node
        while True:
            if current_node.visits <= 0:
                self.rollout(current_node)
            else:
                children = current_node.generate_children()
                if np.random.uniform(0, 1) < self.epsilon:
                        current_node = np.random.choice(children)
                if self.multiplier == 1:
                    current_node = max(children, key=lambda node: node.ucb1)
                else:
                    current_node = min(children, key=lambda node: node.ucb1)
                        
        pass

    def tree_search(self, current_node: Node):
        for i in range(self.M):
            # Assume for now that best_action in game.get_legal_actions(current_node.state) == True
            if current_node.is_leaf_node():
                # If we're at a leaf node, then check whether or not we've visited it
                if current_node.visits >= 1:
                    # If we have, then we do rollout immediately
                    self.rollout(current_node)
                else:
                    # If not visited, then we first generate all children nodes of current node
                    child_nodes = current_node.generate_children()
                    # Now that all children are generated, we need to find the best action to take
                    action_distribution = self.model(current_node.state).numpy()
                    best_action = np.argmax(action_distribution)
                    new_state = (best_action)
            else:
                pass

