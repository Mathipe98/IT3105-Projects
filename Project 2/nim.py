import numpy as np
import copy
from typing import List, Tuple
from node import Node

class Nim:

    def __init__(self, n: int=10, k: int=3) -> None:
        self.n = n
        self.k = k

    def reset(self, player: int) -> Node:
        max_player = True if player == 1 else False
        current_node = Node(state=np.array([self.n]), max_player=max_player)
        current_node.visits = 1
        return current_node
    
    def get_action_space(self) -> int:
        return self.k
    
    def get_legal_actions(self, node: Node) -> List:
        return [i for i in range(1, min(self.k+1, node.state[0]+1))]
    
    def perform_action(self, root_node: Node, action: int, keep_children: bool=False) -> Node:
        # If the root node has a child with the action present, then return that child
        child = next((c for c in root_node.children if c.incoming_edge == action), None)
        if child is not None:
            return child
        # Get the next state
        next_state = root_node.state - action
        # Now create a node with the opposite "parity"
        next_node = Node(
            state=next_state,
            parent=root_node,
            incoming_edge=action,
            max_player=not root_node.max_player
        )
        reward, final = self.evaluate(next_node)
        if final:
            next_node.final = final
            next_node.value = reward
            next_node.visits += 1
        if keep_children:
            root_node.children.append(next_node)
        return next_node
    
    def evaluate(self, node: Node) -> Tuple[int, bool]:
        reward = 0
        final = False
        if self.is_winning(node):
            reward = 1
            final = True
        elif self.is_losing(node):
            reward = -1
            final = True
        # If minimizing player, then flip the reward sign
        if not node.max_player:
            reward *= -1
        return reward, final
    
    def is_winning(self, node: Node) -> bool:
        return node.state[0] <= self.k
    
    def is_losing(self, node: Node) -> bool:
        return node.state[0] == 0
    
    def encode_node(self, node: Node) -> np.ndarray:
        """Method to one-hot encode the state of the Nim-game.
        This includes the player number.

        Args:
            node (Node): Node corresponding to current state
            player (int): Integer describing either first or second player

        Returns:
            np.ndarray: One-hot encoding of the state + player
        """
        player = 1 if node.max_player else -1
        state = node.state
        encoded_state = np.zeros(self.n + 1)
        index = state[0] - 1
        encoded_state[index] = 1
        encoded_state[-1] = player
        return encoded_state
    
    def get_encoded_shape(self) -> Tuple:
        return (11,)

    
if __name__ == "__main__":
    test = Nim()