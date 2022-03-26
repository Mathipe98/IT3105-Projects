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

np.random.seed(123)


class MCTSAgent:

    def __init__(self,
                 game: Game,
                 n_sims: int,
                 episodes: int,
                 epsilon: float = 0.1,
                 model_saves: int = 5,
                 use_best_model: bool = True) -> None:
        self.game = game
        self.n_sims = n_sims
        self.episodes = episodes
        self.epsilon = epsilon
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
                filepath = f"./Project 2/models/target_policy_{self.model_saves - 1}"
            else:
                filepath = f"./Project 2/models/target_policy_{np.random.randint(0, self.model_saves)}"
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
            for e in range(self.episodes):
                print(f"Episode: {e}")
                root_node = self.game.reset()
                mc_tree = MonteCarloTree(
                    root_node=root_node, game=self.game, keep_children=True)
                while not root_node.final:
                    for _ in range(self.n_sims):
                        mc_tree.traverse()
                    target = np.zeros(shape=self.game.get_action_space())
                    va_values = [(child.visits, child.incoming_edge)
                                 for child in root_node.children]
                    if va_values is None or len(va_values) == 0:
                        raise ValueError(
                            "ERROR: Visit/Action array is empty in training")
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
                    next_node = self.game.perform_action(
                        root_node, best_action)
                    root_node = next_node
                    mc_tree = MonteCarloTree(
                        root_node, self.game, keep_children=True)
                if root_node.max_player:
                    p1_wins += 1
                else:
                    p2_wins += 1
                if len(replay_buffer) > 100:
                    minibatch = random.sample(
                        replay_buffer, min(len(replay_buffer), 1000))
                    inputs = np.array([tup[0] for tup in minibatch])
                    targets = np.array([tup[1] for tup in minibatch])
                    self.model.fit(x=inputs, y=targets, epochs=1, verbose=0)
                if e % interval == 0:
                    print("Saving weights...")
                    self.model.save_weights(
                        f"./Project 2/models/target_policy_{e//interval}")
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
                action_probs = self.model(encoded).numpy()
                action = np.argmax(action_probs, axis=1)[0] + 1
                next_node = self.game.perform_action(
                    root_node=node, action=action)
                node = next_node
                _, done = self.game.evaluate(node)
            if node.max_player:
                p1_wins += 1
            else:
                p2_wins += 1
        print(f"NN playing stats:\nP1\t{p1_wins}\nP2\t{p2_wins}")


if __name__ == "__main__":
    game = Game(game_implementation=Nim(), player=1)
    agent = MCTSAgent(game=game, n_sims=100,
                      episodes=500, use_best_model=False)
    model_params = {
        "hidden_layers": (100, 50, 25),
        "hl_activations": ('relu', 'relu', 'sigmoid'),
        "output_activation": 'softmax',
        "optimizer": 'Adam',
        "lr": 0.01,
    }
    agent.setup_model(**model_params)
    agent.train()
    agent.play()
