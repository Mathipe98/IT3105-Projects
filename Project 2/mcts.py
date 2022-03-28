from typing import List
import numpy as np
import random
import tensorflow as tf

from collections import deque
from keras.models import Sequential
from tensorflow import keras

from monte_carlo_tree import MonteCarloTree
from network import create_model
from nim import Nim
from game import Game
from hex import Hex
from visualizer import visualize_hex_node_state

np.random.seed(123)


class MCTSAgent:

    def __init__(self,
                 game: Game,
                 M: int,
                 episodes: int,
                 model_name: str,
                 model_saves: int = 5,
                 use_best_model: bool = True) -> None:
        self.game = game
        self.M = M
        self.episodes = episodes
        self.model_name = model_name
        self.model_saves = model_saves
        self.use_best_model = use_best_model
        self.model = None

    def setup_model(self,
                    hidden_layers: tuple,
                    hl_activations: List,
                    output_activation: str,
                    optimizer: str,
                    lr: float) -> None:
        n_input = self.game.get_encoded_shape()[0]
        n_output = self.game.get_action_space()
        self.model = create_model(
            n_input, hidden_layers, hl_activations,
            n_output, output_activation, optimizer, lr
        )

    def load_weights(self):
        if self.model is None:
            raise ValueError("Load weights called before initializing model.")
        try:
            if self.use_best_model:
                filepath = f"./Project 2/models/{self.model_name}_target_policy_{self.model_saves - 1}"
            else:
                filepath = f"./Project 2/models/{self.model_name}_target_policy_{np.random.randint(0, self.model_saves)}"
            self.model.load_weights(filepath)
            print(f"Read model from file, so I do not retrain.")
            done_training = True
        except:
            print(f"Could not read weights from file. Must retrain...")
            done_training = False
        return done_training

    def train(self):
        done_training = self.load_weights()
        if not done_training:
            replay_buffer = deque(maxlen=100000)
            interval = self.episodes // self.model_saves
            p1_wins = 0
            p2_wins = 0
            M = 10
            difference = self.M // 49
            start_eps = 0.3
            end_eps = 0.05
            for e in range(self.episodes):
                print(f"Episode: {e}")
                root_node = self.game.reset()
                mc_tree = MonteCarloTree(
                    root_node=root_node,
                    game=self.game,
                    model=self.model,
                    start_epsilon=start_eps,
                    end_epsilon=end_eps,
                    keep_children=True)
                action_counter = 0
                while not root_node.final:
                    print(
                        f"Actions performed in episode {e}: {action_counter}")
                    epsilon_decay = round((start_eps - end_eps) / M, 4)
                    for _ in range(M):
                        mc_tree.traverse(epsilon_decay)
                    M = min(M + difference, self.M)
                    target = np.zeros(shape=self.game.get_action_space())
                    va_values = [(child.visits, child.incoming_edge)
                                 for child in root_node.children]
                    if va_values is None or len(va_values) == 0:
                        raise ValueError(
                            "ERROR: Visit/Action array is empty in training")
                    factor = 1.0 / sum(va[0] for va in va_values)
                    for visits, action in va_values:
                        target[action] = visits * factor
                    network_input = self.game.encode_node(node=root_node)
                    target_tuple = (network_input, target)
                    replay_buffer.append(target_tuple)
                    best_action = np.argmax(target)
                    next_node = self.game.perform_action(
                        root_node, best_action)
                    root_node = next_node
                    # Prune the tree
                    root_node.parent = None
                    mc_tree = MonteCarloTree(
                        root_node=root_node,
                        game=self.game,
                        model=self.model,
                        start_epsilon=start_eps,
                        end_epsilon=end_eps,
                        keep_children=True)
                    action_counter += 1
                if root_node.max_player:
                    p1_wins += 1
                else:
                    p2_wins += 1
                if e % interval == 0:
                    print("Saving weights...")
                    self.model.save_weights(
                        f"./Project 2/models/{self.model_name}_target_policy_{e//interval}")
                if len(replay_buffer) > 1000:
                    minibatch = random.sample(
                        replay_buffer, min(len(replay_buffer), 10000))
                    inputs = np.array([tup[0] for tup in minibatch])
                    targets = np.array([tup[1] for tup in minibatch])
                    self.model.fit(x=inputs, y=targets, epochs=1, verbose=0)
                action_counter = 0
            print(f"Training stats:\nP1\t{p1_wins}\nP2\t{p2_wins}")

    def play(self) -> None:
        p1_wins = 0
        p2_wins = 0
        for _ in range(10):
            node = self.game.reset()
            _, done = self.game.evaluate(node)
            while not done:
                encoded = self.game.encode_node(node)
                encoded = encoded.reshape(1, -1)
                legal_actions = self.game.get_actions(node)
                action_probs = self.model(encoded).numpy()
                action = np.argmax(action_probs, axis=1)[0]
                if action not in legal_actions:
                    action = np.random.choice(legal_actions)
                next_node = self.game.perform_action(
                    root_node=node, action=action)
                node = next_node
                _, done = self.game.evaluate(node)
            if node.max_player:
                p1_wins += 1
            else:
                p2_wins += 1
            print(f"VISUALIZING GAME {p1_wins + p2_wins}")
            visualize_hex_node_state(node)
        print(f"NN playing stats:\nP1\t{p1_wins}\nP2\t{p2_wins}")


if __name__ == "__main__":
    game = Game(game_implementation=Hex(7), player=1)
    agent = MCTSAgent(
        game=game,
        M=10000,
        episodes=100,
        model_name="HEX_7x7_BEEFY",
        use_best_model=True)
    model_params = {
        "hidden_layers": (250, 200, 100),
        "hl_activations": ('relu', 'relu', 'relu'),
        "output_activation": 'softmax',
        "optimizer": 'Adam',
        "lr": 0.01,
    }
    agent.setup_model(**model_params)
    agent.train()
    agent.play()
