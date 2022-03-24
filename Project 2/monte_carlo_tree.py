import numpy as np

from game import Game
from nim import Nim
from node import Node

np.random.seed(123)


class MonteCarloTree:

    def __init__(self, root_node: Node, game: Game) -> None:
        self.root_node = root_node
        self.game = game
    
    def traverse(self) -> None:
        root = self.root_node
        while True:
            if root.leaf:
                # If we encounter a leaf node, then flip the value since we've now dealt with it
                root.leaf = False
                if root.final or root.visits == 0:
                    self.rollout(root)
                    break
                else:
                    self.generate_children(root)
                    root = np.random.choice(root.children)
                    self.rollout(root)
                    break
            else:
                next_root = self.get_child(root)
                root = next_root

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
        while True:
            reward, done = self.game.evaluate(node)
            if done:
                node.final = True
                node.leaf = True
                node.value = reward
                node.visits += 1
                self.backprop(node)
                break
            actions = self.game.get_actions(node)
            if actions is None or len(actions) == 0:
                raise RuntimeError("ERROR: get actions in rollout returned nothing; probably called on terminal state")
            random_action = np.random.choice(actions)
            node = self.game.perform_action(root_node=node, action=random_action)

    def backprop(self, node: Node) -> None:
        value = node.value
        while True:
            node = node.parent
            if node is None:
                break
            node.value += value
            node.visits += 1

    def get_child(self, node: Node) -> Node:
        """This method calculated the upper confidence bound value of a node,
        where this value will be used to decide the tree policy, mainly the
        policy the MCTS algorithm uses in order to traverse from root node to
        leaf node.
        It calculates Q(s, a) + u(s,a) which corresponds to
        value/visits + C * sqrt(log(N) / n), where N is parent visits, and n is
        current node visits

        Args:
            node (Node): Node whose children we'll choose between

        Raises:
            RuntimeError: If this method is called before the node has children

        Returns:
            Node: Resulting greedy-optimal choice of child node
        """
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
    game = Game(game_implementation=Nim(), player=1)
    # node = game.reset()
    # node.visits = 5
    # mtree.generate_children(node)
    # print(node.children)
    # c = 1
    # for child in node.children:
    #     child.visits += c
    #     c += 1
    # child = mtree.get_child(node)
    # print(child)
    node = game.reset()
    mtree = MonteCarloTree(root_node=node, game=game)
    mtree.traverse()