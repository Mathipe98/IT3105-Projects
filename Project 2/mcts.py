import numpy as np
import tensorflow as tf
import copy
from tensorflow import keras
from keras import Sequential
from collections import deque
from node import Node
from nim import Nim

np.random.seed(123)
class MCTSAgent:

    def __init__(self,
                game: Nim,
                search_games: int,
                episodes: int,
                player: int=1,
                epsilon: float=0.1,
                model: Sequential=None) -> None:
        self.game = game
        self.search_games = search_games
        self.episodes = episodes
        self.starting_player = player
        self.epsilon = epsilon
        self.model = model
        self.debug = 0
    
    def train(self):
        replay_buffer = deque(maxlen=100000)
        p1_win = 0
        p2_win = 0
        for i in range(self.episodes):
            # print(i)
            max_player = True if self.starting_player == 1 else False
            root_node = self.game.reset()
            # count = 0
            while not root_node.is_final():
                # print(f"Count: {count}")
                self.tree_search(root_node, max_player)
                # At this point, the node visit counts should be updated
                target = np.zeros(self.game.get_n_possible_actions())
                child_visits = [child.visits for child in root_node.children]
                norm_factor = 1.0/sum(child_visits)
                parent_actions = [child.parent_action for child in root_node.children]
                for value, action in zip(child_visits, parent_actions):
                    action_index = action - 1
                    target[action_index] = value * norm_factor
                network_input = self.game.encode_state(root_node, self.starting_player)
                target_tuple = (network_input, target)
                replay_buffer.append(target_tuple)
                best_action = np.argmax(target) + 1
                # print(f"Action taken with player {multiplier}: {best_action}")
                root_node, _, _ = self.game.step(best_action)
                max_player = not max_player
                # count += 1
            if max_player:
                p1_win += 1
            else:
                p2_win += 1
        print(f"Stats:\nP1\t{p1_win}\nP2\t{p2_win}")


    def rollout(self, current_node: Node, original_max_player: bool) -> None:
        leaf_node = current_node
        trajectory = [current_node]
        done = False
        rollout_max_player = True if original_max_player else False
        while not done:
            legal_actions = self.game.get_legal_actions(current_node)
            if legal_actions is None or len(legal_actions) == 0:
                print()
            action = np.random.choice(legal_actions)
            # We now simulate the taken action. This means flipping the state of the minmax player
            next_node, reward, done = self.game.simulate_action(action, current_node)
            rollout_max_player = not rollout_max_player
            trajectory.append(next_node)
            if done:
                # If the player that ended up in a final state is not maximizing, then
                # we make sure that the reward sign is flipped
                if not rollout_max_player:
                    reward *= -1
                self.backprop(leaf_node, reward, original_max_player)
            current_node = next_node

    def backprop(self, leaf_node: Node, value: int, max_player: bool) -> None:
        """This method serves as the MCTS backpropagation algorithm.
        For now, this algorithm only backpropagates the rewards from the leaf
        nodes, meaning it discards the nodes generated only from the rollout

        Args:
            leaf_node (Node): The leaf node that started the rollout
            value (int): The value of the terminal state of the rollout
            multiplier (int): Multiplier signifying which player is playing
        """
        while True:
            # if not max_player:
            #     value = -value
            leaf_node.value += value
            leaf_node.visits += 1
            leaf_node = leaf_node.parent
            # max_player = not max_player
            if leaf_node is None:
                break
        
    def tree_search(self, root_node: Node, max_player: bool) -> None:
        for _ in range(self.search_games):
            return self.tree_recursion(root_node, max_player)
            # current_node = root_node
            # if current_node.is_final():
            #     break
            # while True:
            #     if current_node.visits <= 0:
            #         self.rollout(current_node)
            #         break
            #     else:
            #         children = current_node.children
            #         if len(children) == 0:
            #             children = current_node.generate_children()
            #         if self.starting_player == 1:
            #             current_node = max(children, key=lambda node: node.get_ucb1(multiplier))
            #         elif self.starting_player == 2:
            #             current_node = min(children, key=lambda node: node.get_ucb1(multiplier))
            #         else:
            #             raise ValueError("Starting player must be either 1 or 2")
    
    def tree_recursion(self, root_node: Node, max_player: bool) -> None:
        # NOTE: LEAF NODE VISITS TRIGGERS FOR NEW NODE; DOESNT UPDATE IN ROLLOUT/BACKPROP
        # Basically: some error regarding visit-updates in backprop (probably)
        if root_node.visits <= 0:
            return self.rollout(root_node, max_player)
        children = root_node.children
        if len(children) == 0:
            children = root_node.generate_children()
        if max_player:
            root_node = max(children, key=lambda n: n.get_ucb1(max_player=True))
        else:
            root_node = min(children, key=lambda n: n.get_ucb1(max_player=False))
        return self.tree_recursion(root_node, not max_player)

if __name__ == "__main__":
    test_game = Nim(n=10, k=3)
    agent = MCTSAgent(game=test_game, search_games=1000, episodes=100, player=1)
    agent.train()
