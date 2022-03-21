from typing import List
import numpy as np
from __future__ import annotations

class Node:

    def __init__(self, game, state, parent=None, parent_action=None) -> None:
        self.game = game
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children: List[Node] = []
        self.visited = False
        self.value = 0
    
    def is_leaf_node(self) -> bool:
        return not self.visited or len(self.children) == 0

    def generate_children(self) -> List[Node]:
        all_possible_actions = self.game.get_legal_actions(self.state)
        for action in all_possible_actions:
            new_state = self.game.simulate_action(action, self.state)
            child_node = Node(state=new_state, parent=self, parent_action=action)
            self.children.append(child_node)
        return self.children