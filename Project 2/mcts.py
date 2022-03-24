import numpy as np
# import tensorflow as tf

from collections import deque

from monte_carlo_tree import MonteCarloTree
from nim import Nim
from game import Game

np.random.seed(123)


class MCTSAgent:

    def __init__(self,
                game: Game,
                n_sims: int,
                episodes: int,
                epsilon: float=0.1,
                model=None) -> None:
        self.game = game
        self.n_sims = n_sims
        self.episodes = episodes
        self.epsilon = epsilon
        self.model = model
        self.debug = 0
    
    def train(self):
        replay_buffer = deque(maxlen=100000)
        p1_win = 0
        p2_win = 0
        for k in range(self.episodes):
            if k == 81:
                print()
            root_node = self.game.reset()
            mc_tree = MonteCarloTree(root_node=root_node, game=self.game, keep_children=True)
            while not root_node.final:
                for _ in range(self.n_sims):
                    mc_tree.traverse()
                # if root_node.state[0] == 4 and len(root_node.children) == 0:
                #     print(k)
                #     pass
                target = np.zeros(shape=self.game.get_action_space())
                va_values = [(child.visits, child.incoming_edge) for child in root_node.children]
                if va_values is None or len(va_values) == 0:
                    raise ValueError("ERROR: Visit/Action array is empty in training")
                factor = 1.0 / sum(va[0] for va in va_values)
                # Assume that all actions have integer number ranging from 1 to the action space
                for visits, action in va_values:
                    a_index = action - 1
                    target[a_index] = visits * factor
                network_input = self.game.encode_node(node=root_node)
                target_tuple = (network_input, target)
                replay_buffer.append(target_tuple)
                # a_index = action - 1 => action = index + 1
                best_action = np.argmax(target) + 1
                next_node = self.game.perform_action(root_node, best_action)
                root_node = next_node
            if root_node.max_player:
                p1_win += 1
            else:
                p2_win += 1
        print(f"Stats:\nP1\t{p1_win}\nP2\t{p2_win}")

if __name__ == "__main__":
    game = Game(game_implementation=Nim(), player=1)
    agent = MCTSAgent(game=game, n_sims=100, episodes=100)
    agent.train()
