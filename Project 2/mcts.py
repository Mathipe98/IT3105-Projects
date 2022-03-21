import numpy as np
import tensorflow as tf
import copy
from tensorflow import keras
from keras import Sequential
from collections import deque
from node import Node
from nim import Nim

class MCTSAgent:

    def __init__(self, game: Nim, M: int, model: Sequential) -> None:
        self.game = game
        self.M = M
        self.tree = {}
        self.leaf_nodes = []
    
    
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
    
    def get_child_node(self, parent_node: Node, action: int) -> Node:
        pass

    def rollout(self, current_node: Node) -> None:
        copy_game = copy.copy(self.game)
        trajectory = [current_node]
        done = False
        while not done:
            action_distribution = self.model(current_node.state).numpy()
            # Do something with the legal actions to cut out the illegal actions that the network returns
            legal_actions = self.game.get_legal_actions(current_node.state)
            action = np.argmax(action_distribution)
            next_state, reward, done = copy_game.step(action)


    def tree_search(self, current_node: Node):
        copy_game = copy.copy(self.game)
        for i in range(self.M):
            # Assume for now that best_action in game.get_legal_actions(current_node.state) == True
            if current_node.is_leaf_node():
                # If we're at a leaf node, then check whether or not we've visited it
                if not current_node.visited:
                    # If we have, then we do rollout immediately
                    self.rollout(current_node)
                else:
                    # If not visited, then we first generate all children nodes of current node
                    child_nodes = current_node.generate_children()
                    # Now that all children are generated, we need to find the best action to take
                    action_distribution = self.model(current_node.state).numpy()
                    best_action = np.argmax(action_distribution)
                    new_state = copy_game.step(best_action)

