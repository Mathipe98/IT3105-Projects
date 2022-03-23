from __future__ import annotations
from ctypes import Union
from typing import List
import numpy as np

class Node:

    def __init__(self,
                game,
                state,
                parent: Node=None,
                parent_action=None) -> None:
        self.game = game
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children: List[Node] = []
        self.visits = 0
        self.value = 0
        self.c = 2
        self.ucb1 = np.inf
    
    def get_ucb1(self, max_player: bool) -> float:
        """This method calculated the upper confidence bound value of a node,
        where this value will be used to decide the tree policy, mainly the
        policy the MCTS algorithm uses in order to traverse from root node to
        leaf node.
        It calculates Q(s, a) + u(s,a) which corresponds to
        value/visits + C * sqrt(log(N) / n), where N is parent visits, and n is
        current node visits

        Returns:
            float: The resulting value Q(s,a) + u(s,a)
        """
        if self.visits == 0:
            return np.inf
        avg_val = self.value / self.visits
        parent_visits = self.parent.visits
        root = np.sqrt(np.log(parent_visits) / self.visits)
        if max_player:
            self.ucb1 = avg_val + self.c * root
        else:
            self.ucb1 = avg_val - self.c * root
        return self.ucb1
    
    def get_reward(self) -> int:
        return self.game.evaluate_state(self)
    
    def is_leaf_node(self) -> bool:
        return len(self.children) == 0
    
    def is_final(self) -> bool:
        return self.game.is_winning(self) or self.game.is_losing(self)

    def generate_children(self) -> List[Node]:
        possible_actions = self.game.get_legal_actions(self)
        if self.children is not None and len(possible_actions) == len(self.children):
            return self.children
        for action in possible_actions:
            child_node, _, _ = self.game.simulate_action(action, self)
            self.children.append(child_node)
        assert all(child.parent == self for child in self.children), "Generated child nodes do not have correct parent"
        return self.children
    
    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"State: {self.state}. Val: {self.val}. Visits: {self.visits}."