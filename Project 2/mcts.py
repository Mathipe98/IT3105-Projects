from collections import deque
from nim import Nim
import numpy as np
from game import Game




np.random.seed(123)
class MCTSAgent:

    def __init__(self,
                game: Game,
                search_games: int,
                episodes: int,
                player: int=1,
                epsilon: float=0.1,
                model=None) -> None:
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
        for _ in range(self.episodes):
            actual_player = True if self.starting_player == 1 else False
            max_player = True if actual_player else False
            root_node = self.game.reset()
            while not root_node.is_final():
                self.tree_search(root_node, max_player)
                # At this point, the node visit counts should be updated
                target = np.zeros(self.game.get_action_space())
                child_visits = [child.visits for child in root_node.children]
                norm_factor = 1.0/sum(child_visits)
                parent_actions = [child.incoming_edge for child in root_node.children]
                for value, action in zip(child_visits, parent_actions):
                    action_index = action - 1
                    target[action_index] = value * norm_factor
                network_input = self.game.encode_node(root_node, self.starting_player)
                target_tuple = (network_input, target)
                replay_buffer.append(target_tuple)
                best_action = np.argmax(target) + 1
                # print(f"Action taken with player {1 if max_player else 2}: {best_action}")
                root_node, _, _ = self.game.step(best_action)
                max_player = not max_player
            if max_player:
                p1_win += 1
            else:
                p2_win += 1
        print(f"Stats:\nP1\t{p1_win}\nP2\t{p2_win}")

if __name__ == "__main__":
    test_game = Nim(n=10, k=3)
    agent = MCTSAgent(game=test_game, search_games=100, episodes=100, player=1)
    agent.train()
