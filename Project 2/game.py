import numpy as np

from ctypes import Union
from typing import List, Tuple

from hex import Hex
from nim import Nim
from node import Node

class Game:

    def __init__(self, game_implementation: Nim, player: int=1) -> None:
        self.game_implementation = game_implementation
        self.player = player
    
    def reset(self) -> Node:
        return self.game_implementation.reset(self.player)
    
    def perform_action(self, root_node: Node, action: int, keep_children: bool=False) -> Node:
        return self.game_implementation.perform_action(root_node, action, keep_children)

    def get_action_space(self) -> int:
        return self.game_implementation.get_action_space()
    
    def get_actions(self, node: Node) -> List:
        return self.game_implementation.get_legal_actions(node)
    
    def evaluate(self, node: Node) -> Tuple[int, bool]:
        return self.game_implementation.evaluate(node)
    
    def encode_node(self, node: Node) -> np.ndarray:
        return self.game_implementation.encode_node(node)    

    def get_encoded_shape(self) -> Tuple:
        return self.game_implementation.get_encoded_shape()            
    

if __name__ == "__main__":
    game = Game(game_implementation=Hex())
    root_node = game.reset()
    actions = game.get_actions(root_node)
    print(actions)
    next_node = game.perform_action(root_node=root_node, action=actions[-1], keep_children=True)