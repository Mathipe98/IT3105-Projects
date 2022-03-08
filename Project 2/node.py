from sre_parse import State
import numpy as np


class Node:

    def __init__(self, state, parent=None, a_parent=None) -> None:
        self.state = State
        self.parent = parent
        self.a_parent = a_parent
        self.children = []
        self.n_visits = 0