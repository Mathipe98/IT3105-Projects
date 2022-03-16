from typing import List
import numpy as np


class Node:

    def __init__(self, state, parent=None, parent_action=None) -> None:
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children: List[Node] = []
        self.visited = False
        self.value = 0