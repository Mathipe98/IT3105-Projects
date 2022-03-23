from __future__ import annotations
from ctypes import Union
from typing import List
import numpy as np

class Node:

    def __init__(self,
                state,
                parent: Node=None,
                incoming_edge: int=None,
                # REMEMBER TO SET MAX PLAYER
                max_player: bool=True,
                leaf: bool=True) -> None:
        # Values from instantiation
        self.state = state
        self.parent = parent
        self.incoming_edge = incoming_edge
        self.max_player = max_player
        self.leaf = leaf

        # Default values
        self.children: List[Node] = []
        self.visits = 0
        self.value = 0
        self.final = None
    
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
        c = 2
        avg_val = self.value / self.visits
        parent_visits = self.parent.visits
        root = np.sqrt(np.log(parent_visits) / self.visits)
        if max_player:
            ucb1 = avg_val + c * root
        else:
            ucb1 = avg_val - c * root
        return ucb1
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"\nState: {self.state}\nVal: {self.value}\nVisits: {self.visits} \
            \nParent state: {self.parent.state}\nIncoming edge: {self.incoming_edge}\n"