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

    """
    REMEMBER:
    In DFS, if I just check the top row for 1's, and then
    iterate downward and don't find another 1, then I can just
    return False because then I know nothing is connected.
    Or something along those lines.
    Just think if I have [1,0,0,1], [1,0,0,1], [0,0,0,1],
    then I have to take into account that the first 1 leads
    to nothing, but the last 1 leads to an actual path.
    I'll figure something out.
    """
    
    def connected_top_bottom(self, node: Node, target: int) -> bool:
        visited = []
        queue = []
        # Append all indexes corresponding to top and bottom row
        queue.extend(
            [(0, i) for i in range(self.board_size)] +
            [(self.board_size - 1, i) for i in range(self.board_size)]
        )
        for i in range(self.board_size):
            if node.state[(0, i)] != target:
                continue
            queue.append((0, i))
            if node.state[(self.board_size-1, i)] != target:
                continue
            queue.append((self.board_size-1, i))
        while queue:
            row, col = queue.pop(0)
            visited.append((row, col))
            for neighbour in self.get_neighbours((row, col)):
                if neighbour not in visited:
                    if node.state[neighbour] == target:
                        queue.append(neighbour)
                        visited.append(neighbour)
        if len(visited) == 0:
            return False
        # Remove duplicates
        visited = list(set(visited))
        # Sort visited nodes by row
        visited.sort(key=lambda coords: coords[0])
        while True:
            row, col = visited.pop(0)
            if row == self.board_size - 1:
                return True
            next_coords = next((c for c in visited if c[0] == row + 1), None)
            if next_coords is None:
                return False
    
    def bfs(self, node: Node) -> bool:
        
        # Append all indexes corresponding to top and bottom row
        left_col = [(i, 0) for i in range(self.board_size)]
        right_col = [(i, self.board_size - 1) for i in range(self.board_size)]
        queue.extend(left_col + right_col)
        while queue:
            row, col = queue.pop(0)
            if node.state[row,col] != target:
                continue
            visited.append((row, col))
            for neighbour in self.get_neighbours((row, col)):
                if neighbour not in visited:
                    if node.state[neighbour] == target:
                        queue.append(neighbour)
                        visited.append(neighbour)
        if len(visited) == 0:
            return False
        # Remove duplicates
        visited = list(set(visited))
        # Sort visited nodes by col
        visited.sort(key=lambda coords: coords[1])
        while True:
            row, col = visited.pop(0)
            if col == self.board_size - 1:
                return True
            next_coords = next((c for c in visited if c[1] == col + 1), None)
            if next_coords is None:
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

if __name__ == '__main__':
    test = Hex()
    test_state = np.zeros(shape=(7,7))
    # test_state[0,0] = 2
    # test_state[1,0] = 2
    # test_state[2,0] = 2
    # test_state[2,1] = 2
    # test_state[3,1] = 2
    # test_state[3,2] = 2
    # test_state[3,3] = 2
    # test_state[3,4] = 2
    # test_state[3,5] = 2
    # test_state[4,5] = 2
    # test_state[4,6] = 2
    test_state[:,0] = 2
    test_state[:,6] = 2
    test_state[3,:] = 2
    test_state[3,3] = 2
    print(test_state)
    test_node = Node(state=test_state, max_player=True)
    print(test.bfs(test_node))
