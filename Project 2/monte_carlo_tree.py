import numpy as np
from node import Node
from nim import Nim
from game import Game


class MonteCarloTree:

    def __init__(self, root_node: Node, game: Game) -> None:
        self.root_node = root_node
        self.game = game
    
    def traverse(self) -> None:
        if self.root_node.leaf:
            pass
        else:
            pass

    def rollout(self, node: Node) -> None:
        """
        While True:
            reward, done = self.game.evaluate(node)
            if done:
                return reward
            action = np.random.choice(self.game.get_actions(node))
            node = self.game.perform_action(action)

        Args:
            node (Node): _description_
        """
        pass

    def backprop(self, node: Node) -> None:
        pass

    def get_child(self, node: Node) -> Node:
        c = 2
        children = node.children
        if children is None:
            raise RuntimeError("Called get_child when node has no children.")
        child_ucb_vals = []
        for child in children:
            if child.visits == 0:
                child_ucb_vals.append(np.inf)
                continue
            Q = child.value / child.visits
            root = np.sqrt(np.log(node.visits) / child.visits)
            if node.max_player:
                child_ucb_vals.append(Q + c * root)
            else:
                child_ucb_vals.append(Q - c * root)
        if node.max_player:
            return children[np.argmax(child_ucb_vals)]
        return children[np.argmin(child_ucb_vals)]
    
    def generate_children(self, node: Node) -> None:
        # Get possible actions for current node
        actions = self.game.get_actions(node)
        for action in actions:
            # If a previous action already produced a child, then don't regenerate the child
            if any(c.incoming_edge == action for c in node.children):
                actions.remove(action)
        # Now generate the children with actions that have not been taken
        for action in actions:
            child_node = self.game.perform_action(root_node=node, action=action)
            assert child_node.parent==node and child_node.incoming_edge==action, "Action performed did not properly update logic"
            node.children.append(child_node)
        assert all(c.parent==node for c in node.children), "Children update incorrectly"

if __name__ == "__main__":
    game = Game(game_implementation=Nim(), player=2)
    mtree = MonteCarloTree(None, game)
    node = game.reset()
    node.visits = 5
    mtree.generate_children(node)
    print(node.children)
    c = 1
    for child in node.children:
        child.visits += c
        c += 1
    child = mtree.get_child(node)
    print(child)