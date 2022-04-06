from re import T
import numpy as np
import copy

from typing import List, Tuple
from node import Node
from visualizer import visualize_hex_node_state


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
        new_state[row_index, column_index] = 1 if root_node.max_player else -1
        next_node = Node(
            state=new_state,
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
        elif self.is_draw(node):
            reward = 0
            final = True
        if not node.max_player:
            reward *= -1
        return reward, final

    def is_winning(self, node: Node) -> bool:
        if node.max_player:
            target = 1
            return self.connected_top_bottom(node, target)
        target = -1
        return self.connected_left_right(node, target)
    
    def is_losing(self, node: Node) -> bool:
        # P2 is winning <=> P1 is losing
        if node.max_player:
            target = -1
            return self.connected_left_right(node, target)
        target = 1
        return self.connected_top_bottom(node, target)
    
    def is_draw(self, node: Node) -> bool:
        return not self.is_winning(node) and not self.is_losing(node) and \
            len(self.get_legal_actions(node)) == 0
    
    def connected_top_bottom(self, node: Node, target: int) -> bool:
        visited = []
        queue = []
        # Append all indexes corresponding to top row
        queue.extend(
            [(0, i) for i in range(self.board_size) if node.state[(0,i)] == target]
        )
        if len(queue) == 0:
            return False
        while queue:
            coords = queue.pop(0)
            visited.append(coords)
            for neighbour in self.get_neighbours(coords):
                if neighbour not in visited:
                    if node.state[neighbour] == target and neighbour not in queue:
                        queue.append(neighbour)
        if len(visited) == 0:
            return False
        if any(c[0] == self.board_size-1 for c in visited):
            return True
        return False
    
    def connected_left_right(self, node: Node, target: int) -> bool:
        visited = []
        queue = []
        # Append all indexes corresponding to left column
        queue.extend(
            [(i, 0) for i in range(self.board_size) if node.state[(i,0)] == target]
        )
        if len(queue) == 0:
            return False
        while queue:
            coords = queue.pop(0)
            visited.append(coords)
            for neighbour in self.get_neighbours(coords):
                if neighbour not in visited:
                    if node.state[neighbour] == target and neighbour not in queue:
                        queue.append(neighbour)
        if len(visited) == 0:
            return False
        if any(c[1] == self.board_size-1 for c in visited):
            return True
        return False
    
    def get_neighbours(self, indeces: Tuple) -> List[Tuple]:
        row, col = indeces
        max_col = min(col + 1, self.board_size - 1)
        min_col = max(0, col - 1)
        max_row = min(row + 1, self.board_size - 1)
        min_row = max(0, row - 1)
        return [
            (min_row, col),
            (min_row, max_col),
            (row, max_col),
            (max_row, col),
            (max_row, min_col),
            (row, min_col)
        ]
    
    def encode_node(self, node: Node) -> np.ndarray:
        player = 1 if node.max_player else -1
        state = node.state.flatten()
        encoded_state = np.append(state, player)
        return encoded_state
    
    def get_encoded_shape(self) -> Tuple:
        state_shape = self.reset(1).state.flatten().shape
        encoded_shape = (state_shape[0] + 1,)
        return encoded_shape
    

def play_random() -> None:
    game = Hex(board_size=10)
    node = game.reset(player=1)
    print(game.get_encoded_shape())
    while not node.final:
        actions = game.get_legal_actions(node)
        action = np.random.choice(actions)
        node = game.perform_action(root_node=node, action=action)
    print(f"Final game state:\n{node}")
    print(f"P1 wins?\t{not node.max_player and game.is_losing(node)}\nP2 wins?\t{node.max_player and game.is_losing(node)}")
    print(f"Final node evaluation:\t {game.evaluate(node)}")
    visualize_hex_node_state(node)

if __name__ == '__main__':
    play_random()
    # g = Hex()
    # n = g.reset(1)
    # print(g.get_legal_actions(n))
