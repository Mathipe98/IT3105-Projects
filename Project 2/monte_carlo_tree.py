import copy
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras import Sequential

from game import Game
from hex import Hex
from node import Node

np.random.seed(123)


class MonteCarloTree:

    def __init__(self, root_node: Node, game: Game, model: Sequential,
                model_is_trained: bool, epsilon: float=0.1, keep_children: bool=False) -> None:
        self.root_node = root_node
        self.game = game
        self.keep_children = keep_children
        self.model = model
        self.model_is_trained = model_is_trained
        self.epsilon = epsilon
    
    def traverse(self) -> None:
        """This method implements the tree traversal part of
        MCTS. It starts with the trees own root node, and 
        iterates in a loop until it finds a leaf node to either
        rollout or expand.
        """
        root = self.root_node
        while True:
            if root.leaf:
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
        """This method performs the rollout in Monte-Carlo
        Tree Search. It takes a leaf node, and chooses either
        random actions or actions determined by a neural network,
        and performs them until a terminal state is reached.

        Args:
            node (Node): Leaf node with which to start rolling out

        Raises:
            RuntimeError: Error in case rollout is called on a terminal state
        """
        while True:
            if node.final:
                self.backprop(node)
                break
            actions = self.game.get_actions(node)
            if actions is None or len(actions) == 0:
                raise RuntimeError("ERROR: get actions in rollout returned nothing; probably called on terminal state")
            # Greedy epsilon + if model is not trained, then choose random action anyway
            if np.random.uniform(0,1) < self.epsilon or not self.model_is_trained:
                action = np.random.choice(actions)
            else:
                network_output = self.model(self.game.encode_node(node).reshape(1,-1)).numpy()
                action = np.argmax(network_output, axis=1)
                # If the network, during the early phases, opts for an illegal action, then choose a random one
                if action not in actions:
                    action = np.random.choice(actions)
            node = self.game.perform_action(root_node=node, action=action, keep_children=self.keep_children)

    def backprop(self, node: Node) -> None:
        """This method takes the value of a terminal node,
        and backpropagates these values up through the tree.

        Args:
            node (Node): Terminal node whose value will backpropagate
        """
        value = np.sign(node.value) * 1
        while True:
            node.visits += 1
            node.value += value
            node = node.parent
            if node is None:
                break

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
        if node.max_player:
            sign = 1
        else:
            sign = -1
        c = 1.0
        children = node.children
        if children is None or len(children) == 0:
            raise RuntimeError("Called get_child when node has no children.")
        child_ucb_vals = []
        for child in children:
            if child.visits == 0:
                child_ucb_vals.append(sign * np.inf)
                continue
            Q = child.value / child.visits
            root = np.sqrt(np.log(node.visits) / child.visits)
            child_ucb_vals.append(Q + sign * c * root)
        if node.max_player:
            child = children[np.argmax(child_ucb_vals)]
        else:
            child = children[np.argmin(child_ucb_vals)]
        return child
    
    def generate_children(self, node: Node) -> None:
        """This method will "expand" a leaf node, meaning
        it will check all actions possible to do from this node,
        and then add the resulting nodes from these actions as
        children of the original node.

        Args:
            node (Node): Node whose children will be generated
        """
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
        assert all(c.parent==node for c in node.children), "Children updates are incorrect"
        # Since we've expanded a node, it's no longer a leaf node
        node.leaf = False
    
    def reset_values(self) -> None:
        root = self.root_node
        root.visits = 1
        queue = copy.copy(root.children)
        while len(queue) > 0:
            for node in queue:
                node.visits = 0
                node.value = np.sign(node.value) if node.final else 0
                if len(node.children) != 0:
                    queue.extend(node.children)
                queue.remove(node)
        
