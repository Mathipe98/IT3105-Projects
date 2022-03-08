import numpy as np


class MCTS:

    def __init__(self) -> None:
        self.tree = {}
        self.leaf_nodes = []
    

    