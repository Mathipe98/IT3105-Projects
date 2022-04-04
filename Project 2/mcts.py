import ast
import copy
from matplotlib import pyplot as plt
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
from node import Node
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
                 tree_traversals: int,
                 n_episodes: int,
                 model_name: str,
                 model_saves: int = 100,
                 force_relearn: bool = False,
                 display_training: bool = False,
                 display_playing: bool = True,
                 topp: bool = False) -> None:
        self.game = game
        self.tree_traversals = tree_traversals
        self.episodes = n_episodes
        self.model_name = model_name
        self.model_saves = model_saves
        self.force_relearn = force_relearn
        self.display_training = display_training
        self.display_playing = display_playing
        self.topp = topp
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
            if self.topp:
                dir = "topp_models"
            else:
                dir = "models"
            filepath = f"./{dir}/{self.model_name}"#_{self.episodes // self.model_saves}"
            self.model.load_weights(filepath)
            print(f"Read model from file!")
            done_training = True
        except:
            print(f"Could not read weights from file... Must train from scratch.")
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
        if self.display_training:
            plt.figure(figsize=(6,6))
            plt.ion()
            plt.show()
            visualize_hex_node_state(root_node)
        while not root_node.final:
            mc_tree.reset_values()
            print(
                f"\nActions performed in episode {episode}: {action_counter}")
            start = time.time()
            for _ in range(self.tree_traversals):
                mc_tree.traverse()
            end = time.time()
            print(f"Time taken for tree traversal:\n\t{end-start}")
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
            if self.display_training:
                visualize_hex_node_state(root_node)
                print(f"Visualized training action #{action_counter}")
                time.sleep(2)
        if self.display_training:
            visualize_hex_node_state(root_node)
            print(f"Visualizing final state in game #{episode}")
            time.sleep(2)
        if episode % interval == 0:
            if self.topp:
                dir = "topp_models"
            else:
                dir = "models"
            print("Saving weights...")
            filepath = f"./{dir}/{self.model_name}_{episode//interval}"
            self.model.save_weights(filepath)
        print("TRAINING NETWORK")
        self.model_is_trained = True
        for _ in range(5):
            minibatch = random.sample(
                self.replay_buffer, min(len(self.replay_buffer), 1000))
            inputs = np.array([tup[0] for tup in minibatch])
            targets = np.array([tup[1] for tup in minibatch])
            self.model.fit(x=inputs, y=targets, epochs=1, verbose=1)
        self.lite_model = LiteModel.from_keras_model(self.model)
        plt.close()
        
    def train(self):
        if not self.force_relearn:
            done_training = self.load_weights()
        if self.force_relearn or not done_training:
            interval = self.episodes // self.model_saves
            self.lite_model = LiteModel.from_keras_model(self.model)
            for e in range(self.episodes):
                self.train_single_episode((e, interval))
    
    def get_human_move(self, from_node: Node) -> Node:
        while True:
            action_tuple = ast.literal_eval(
                input("Type in your move: (row, column):\n"))
            value = from_node.state[action_tuple]
            if value == 0:
                break
            else:
                print("Illegal move; a piece is already there")
        action = action_tuple[0] * self.game.game_implementation.board_size + action_tuple[1]
        next_node = self.game.perform_action(
            root_node=from_node, action=action)
        if self.display_training:
            print(f"State after action {action_tuple}:\n")
            visualize_hex_node_state(next_node)
        return next_node
    
    def get_actor_move(self, from_node: Node) -> Node:
        encoded_node = self.game.encode_node(from_node).reshape(1, -1)
        action_probs = self.model(encoded_node).numpy()
        action = np.argmax(action_probs, axis=1)[0]
        legal_actions = self.game.get_actions(from_node)
        if action not in legal_actions:
            for i in range(2, action_probs.shape[1]):
                # Get i'th largest value from action_probs
                action = np.partition(action_probs[0], -i - 1)[-i - 1]
                if action in legal_actions:
                    break
        if action not in legal_actions:
            action = np.random.choice(legal_actions)
        next_node = self.game.perform_action(
            root_node=from_node, action=action)
        if self.display_playing:
            print(f"State after neural network action:\n")
            visualize_hex_node_state(next_node)
        return next_node

    def play_against_network(self) -> None:
        node = self.game.reset()
        _, done = self.game.evaluate(node)
        if self.display_playing:
            print(f"Starting state:\n")
            visualize_hex_node_state(node)
        starting = input(
            "Do you want to start as player 1? Type 'yes' or 'no'\n")
        if starting == "yes":
            while True:
                next_node = self.get_human_move(node)
                _, done = self.game.evaluate(next_node)
                if done:
                    print("Congratulations! You beat the AI program.")
                    if self.display_playing:
                        visualize_hex_node_state(next_node)
                        print()
                        time.sleep(5)
                    break
                node = next_node
                next_node = self.get_actor_move(node)
                _, done = self.game.evaluate(next_node)
                if done:
                    print("Sorry, you lost against the AI program.")
                    if self.display_playing:
                        visualize_hex_node_state(next_node)
                        print()
                        time.sleep(5)
                    break
                node = next_node
        else:
            while True:
                next_node = self.get_actor_move(node)
                _, done = self.game.evaluate(next_node)
                if done:
                    print("Sorry, you lost against the AI program.")
                    if self.display_playing:
                        visualize_hex_node_state(next_node)
                        print()
                        time.sleep(5)
                    break
                node = next_node
                next_node = self.get_human_move(node)
                _, done = self.game.evaluate(next_node)
                if done:
                    print("Congratulations! You beat the AI program.")
                    if self.display_playing:
                        visualize_hex_node_state(next_node)
                        print()
                        time.sleep(5)
                    break
                node = next_node

    def get_tournament_action(self, encoded_state: List, flip_gamestate: bool) -> Tuple[int, int]:
        # Create a node with the state in order to get legal actions
        size = self.game.game_implementation.board_size
        node_state = np.array(encoded_state[1:]).reshape(size, size)
        node = Node(state=node_state)
        legal_actions = self.game.get_actions(node)
        # Now format the encoded state for input to the network
        encoded_state = np.roll(np.array(encoded_state), -1)
        encoded_state[encoded_state == 2] = -1
        encoded = encoded_state.reshape(1, -1)
        if flip_gamestate:
            # Swap 1 and -1 values with each other
            copy_array = copy.copy(encoded)
            encoded[copy_array == 1] = -1
            encoded[copy_array == -1] = 1
        action_probs = self.model(encoded).numpy()
        action = np.argmax(action_probs, axis=1)[0]
        action_is_legal = action in legal_actions
        if not action_is_legal:
            for i in range(2, action_probs.shape[1]):
                # Get i'th largest value from action_probs
                action = np.partition(action_probs[0], -i - 1)[-i - 1]
                if action in legal_actions:
                    break
        # If we still did not manage to produce a legal action, then choose randomly
        if action not in legal_actions:
            action = np.random.choice(legal_actions)
        row = action // self.game.game_implementation.board_size
        col = action - size * row
        return int(row),int(col)
