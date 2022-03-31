import ast
import numpy as np
import os
import random
import tensorflow as tf
import time

from collections import deque
from keras.models import Sequential
from multiprocessing import Pool
from tensorflow import keras
from typing import List, Tuple

from monte_carlo_tree import MonteCarloTree
from network import create_model, LiteModel
from nim import Nim
from game import Game
from hex import Hex
from visualizer import visualize_hex_node_state

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4' 
np.random.seed(123)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)


class MCTSAgent:

    def __init__(self,
                 game: Game,
                 M: int,
                 episodes: int,
                 model_name: str,
                 model_saves: int = 100,
                 use_best_model: bool = True,
                 force_relearn: bool = False) -> None:
        self.game = game
        self.M = M
        self.episodes = episodes
        self.model_name = model_name
        self.model_saves = model_saves
        self.use_best_model = use_best_model
        self.force_relearn = force_relearn
        self.model = None
        self.lite_model = None
        self.replay_buffer = deque(maxlen=100000)
        self.model_is_trained = False
        self.traversal_times = deque(maxlen=1000)

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
                # np.random.randint(0, self.model_saves)}"
                filepath = f"./Project 2/models/{self.model_name}_target_policy_{13}"
            self.model.load_weights(filepath)
            print(f"Read model from file, so I do not retrain.")
            done_training = True
        except:
            print(f"Could not read weights from file. Must retrain...")
            done_training = False
        return done_training

    def train_single_episode(self, episode_interval_tuple: Tuple) -> None:
        episode = episode_interval_tuple[0]
        interval = episode_interval_tuple[1]
        root_node = self.game.reset()
        mc_tree = MonteCarloTree(
            root_node=root_node,
            game=self.game,
            model=self.lite_model,
            model_is_trained=self.model_is_trained,
            keep_children=True)
        action_counter = 0
        while not root_node.final:
            mc_tree.reset_values()
            print(
                f"\nActions performed in episode {episode}: {action_counter}")
            start = time.time()
            for _ in range(self.M):
                mc_tree.traverse()
            end = time.time()
            print(f"Time taken for tree traversal:\n\t{end-start}")
             # M = min(M + difference, self.M)
            self.traversal_times.append(end-start)
            print(f"Average time taken for tree traversal the last 1000 iterations:\n\t{np.mean(self.traversal_times)}")
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
            self.replay_buffer.append(target_tuple)
            best_action = np.argmax(target)
            next_node = self.game.perform_action(
                root_node, best_action)
            root_node = next_node
            # Prune the tree
            root_node.parent = None
            mc_tree = MonteCarloTree(
                root_node=root_node,
                game=self.game,
                model=self.lite_model,
                model_is_trained=self.model_is_trained,
                keep_children=True)
            action_counter += 1
        if len(self.replay_buffer) > 1000:
            if episode % interval == 0:
                print("Saving weights...")
                self.model.save_weights(
                f"./Project 2/models/{self.model_name}_{episode//interval}")
            print("TRAINING NETWORK")
            self.model_is_trained = True
            minibatch = random.sample(
                self.replay_buffer, 1000)
            inputs = np.array([tup[0] for tup in minibatch])
            targets = np.array([tup[1] for tup in minibatch])
            self.model.fit(x=inputs, y=targets, epochs=1, verbose=1)
            self.lite_model = LiteModel.from_keras_model(self.model)
        
    def train(self):
        if not self.force_relearn:
            done_training = self.load_weights()
        if self.force_relearn or not done_training:
            interval = self.episodes // self.model_saves
            self.lite_model = LiteModel.from_keras_model(self.model)
            for e in range(self.episodes):
                self.train_single_episode((e, interval))
                # Start with M = 2000, then reduce when neural network actually enters
                if self.model_is_trained:
                    self.M = 1000

    def play_against_network(self) -> None:
        node = self.game.reset()
        _, done = self.game.evaluate(node)
        print(f"Starting state:\n")
        visualize_hex_node_state(node)
        starting = input(
            "Do you want to start as player 1? Type 'yes' or 'no'\n")
        if starting == "yes":
            while True:
                while True:
                    action_tuple = ast.literal_eval(
                        input("Type in your move: (row, column):\n"))
                    value = node.state[action_tuple]
                    if value == 0:
                        break
                    else:
                        print("Illegal move; a piece is already there")()
                action = action_tuple[0] * self.game.game_implementation.board_size + action_tuple[1]
                next_node = self.game.perform_action(
                    root_node=node, action=action)
                print(f"State after action {action_tuple}:\n")
                visualize_hex_node_state(next_node)
                _, done = self.game.evaluate(next_node)
                if done:
                    print("Congratulations! You beat the AI program.")
                    break
                node = next_node
                encoded_node = self.game.encode_node(node).reshape(1, -1)
                action_probs = self.model(encoded_node).numpy()
                action = np.argmax(action_probs, axis=1)[0]
                legal_actions = self.game.get_actions(node)
                if action not in legal_actions:
                    action = np.random.choice(legal_actions)
                next_node = self.game.perform_action(
                    root_node=node, action=action)
                print(f"State after neural network action:\n")
                visualize_hex_node_state(next_node)
                _, done = self.game.evaluate(next_node)
                if done:
                    print("Sorry, you lost against the AI program.")
                    break
                node = next_node
        else:
            while True:
                encoded_node = self.game.encode_node(node).reshape(1, -1)
                action_probs = self.model(encoded_node).numpy()
                action = np.argmax(action_probs, axis=1)[0]
                legal_actions = self.game.get_actions(node)
                if action not in legal_actions:
                    action = np.random.choice(legal_actions)
                next_node = self.game.perform_action(
                    root_node=node, action=action)
                print(f"State after neural network action:\n")
                visualize_hex_node_state(next_node)
                _, done = self.game.evaluate(next_node)
                if done:
                    print("Sorry, you lost against the AI program.")
                    break
                node = next_node
                while True:
                    action_tuple = ast.literal_eval(
                        input("Type in your move: (row, column):\n"))
                    value = node.state[action_tuple]
                    if value == 0:
                        break
                    else:
                        print("Illegal move; a piece is already there")()
                action = action_tuple[0] * self.game.game_implementation.board_size + action_tuple[1]
                next_node = self.game.perform_action(
                    root_node=node, action=action)
                print(f"State after action {action_tuple}:\n")
                visualize_hex_node_state(next_node)
                _, done = self.game.evaluate(next_node)
                if done:
                    print("Congratulations! You beat the AI program.")
                    break
                node = next_node


if __name__ == "__main__":
    game = Game(game_implementation=Hex(7), player=1)
    agent = MCTSAgent(
        game=game,
        M=2000,
        episodes=1000,
        model_name="HEX_7x7_large-dims_kl-div_sigmoid-softmax",
        use_best_model=False,
        force_relearn=True)
    model_params = {
        "hidden_layers": (512, 256,),
        "hl_activations": ('relu', 'sigmoid'),
        "output_activation": 'softmax',
        "optimizer": 'Adam',
        "lr": 0.01,
    }
    agent.setup_model(**model_params)
    agent.train()
    agent.play_against_network()
