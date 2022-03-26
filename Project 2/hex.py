import numpy as np
import copy

from typing import List, Tuple
from node import Node


class Hex:

    def __init__(self, board_size: int=7) -> None:
        self.board_size = board_size
        
    def reset(self, player: int) -> Node:
        state = np.zeros(shape=(self.board_size, self.board_size))
        node = Node(state=state, max_player=True if player == 1 else False)
        node.visits += 1
        return node
    
    def get_action_space(self) -> int:
        return self.board_size ** 2
    
    def get_legal_actions(self, node: Node) -> List:
        actions = []
        state = node.state
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i,j] == 0:
                    action = j + i * state.shape[0]
                    actions.append(action)
        return actions
    
    def perform_action(self, root_node: Node, action: int, keep_children: bool=False) -> Node:
        child = next((c for c in root_node.children if c.incoming_edge == action), None)
        if child is not None:
            return child
        row_index = action // self.board_size
        column_index = action - row_index * self.board_size
        value_at_index = root_node.state[row_index, column_index]
        if value_at_index != 0:
            raise ValueError("Attempted to place piece where another one already lies")
        new_state = copy.copy(root_node.state)
        new_state[row_index, column_index] = 1 if root_node.max_player else 2
        next_node = Node(
            state=new_state,
            parent=root_node,
            incoming_edge=action,
            max_player=not root_node.max_player
        )
        # reward, final = self.evaluate(next_node)
        reward, final = 0, False
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
        if not node.max_player:
            reward *= -1
        return reward, final

    def is_winning(self, node: Node) -> bool:
        
        return node.state[0] <= self.k
    
    def is_losing(self, node: Node) -> bool:
        return node.state[0] == 0
    
