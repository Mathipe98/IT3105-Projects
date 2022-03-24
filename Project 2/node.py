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
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return str(self.state)
        # return f"\nState: {self.state}\nVal: {self.value}\nVisits: {self.visits} \
        #     \nParent state: {self.parent.state}\nIncoming edge: {self.incoming_edge}\n"