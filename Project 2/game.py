from ctypes import Union
from typing import List, Tuple
import numpy as np
from node import Node
from nim import Nim

class Game:

    def __init__(self, game_implementation: Nim, player: int=1) -> None:
        self.game_implementation = game_implementation
        self.player = player
    
    def reset(self) -> Node:
        return self.game_implementation.reset(self.player)
    
    def perform_action(self, root_node: Node, action: int) -> Node:
        return self.game_implementation.perform_action(root_node, action)

    def get_action_space(self) -> int:
        return self.game_implementation.get_action_space()
    
    def get_actions(self, node: Node) -> List:
        return self.game_implementation.get_legal_actions(node)
    
    def evaluate(self, node: Node) -> Tuple[int, bool]:
        reward = 0
        done = False
        if self.game_implementation.is_winning(node):
            reward = 1
            done = True
        elif self.game_implementation.is_losing(node):
            reward = -1
            done = True
        return reward, done
    
    def encode_node(self, node: Node) -> np.ndarray:
        return self.game_implementation.encode_node(node)                
    

if __name__ == "__main__":
    game = Game(game_implementation=Nim())
    root_node = game.reset()
    actions = game.get_actions(root_node)
    action = np.random.choice(actions)
    new_node = game.perform_action(root_node, action)
    print(f"New node: {new_node}")