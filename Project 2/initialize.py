import numpy as np
import ast
import configparser
import os
import pathlib
import tensorflow as tf

from typing import Dict, Tuple

from ActorClient import ActorClient
from game import Game
from mcts import MCTSAgent
from node import Node
from hex import Hex, visualize_hex_node_state

os.chdir(pathlib.Path(__file__).parent.resolve())

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
np.random.seed(123)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)


def parse_config() -> Tuple[Dict, Dict, Dict]:
    config = configparser.ConfigParser()
    config.read('./config.txt')
    actor_config = {}
    game_config = {}
    network_config = {}
    for section in config.sections():
        if section == "ACTOR":
            actor_config["n_episodes"] = int(
                config.get(section, "n_episodes"))
            actor_config["tree_traversals"] = int(
                config.get(section, "tree_traversals"))
            actor_config["display_training"] = ast.literal_eval(
                config.get(section, "display_training")
            )
            actor_config["display_playing"] = ast.literal_eval(
                config.get(section, "display_playing")
            )
            actor_config["force_relearn"] = ast.literal_eval(
                config.get(section, "force_relearn")
            )
            actor_config["model_name"] = str(config.get(section, "model_name"))
            actor_config["model_saves"] = int(
                config.get(section, "model_saves"))
            actor_config["topp"] = ast.literal_eval(
                config.get(section, "topp")
            )
            actor_config["topp_games"] = int(
                config.get(section, "topp_games"))
        elif section == "GAME":
            game_config["board_size"] = int(config.get(section, "board_size"))
        elif section == "NETWORK":
            network_config["hidden_layers"] = ast.literal_eval(
                config.get(section, "hidden_layers"))
            network_config["hl_activations"] = ast.literal_eval(
                config.get(section, "hl_activations"))
            network_config["output_activation"] = str(
                config.get(section, "output_activation"))
            network_config["optimizer"] = str(config.get(section, "optimizer"))
            network_config["lr"] = float(
                config.get(section, "lr"))
    verify_configs(actor_config, game_config, network_config)
    return actor_config, game_config, network_config


def verify_configs(actor_config: Dict, game_config: Dict, network_config: Dict) -> None:
    assert actor_config["n_episodes"] > 0, "n_episodes must be greater than 0"
    assert actor_config["tree_traversals"] > 0, "tree_traversals must be greater than 0"
    assert actor_config["topp_games"] > 0, "topp_games must be greater than 0"
    assert 11 > game_config["board_size"] > 2, "board_size must be between 3 and 10"
    assert len(network_config["hidden_layers"]) > 0, "hidden_layers must be a tuple of positive integers"
    assert len(network_config["hl_activations"]) == len(network_config["hidden_layers"]), "hl_activations must be a tuple of strings of length equal to hidden_layers"
    for activation in network_config["hl_activations"]:
        assert activation in ["relu", "sigmoid", "tanh", "linear"], "hl_activations must be one of 'relu', 'sigmoid', 'tanh', or 'linear'"
    assert network_config["output_activation"] in ["relu", "sigmoid", "tanh", "linear", "softmax"], "output_activation must be one of 'relu', 'sigmoid', 'tanh', 'linear', or 'softmax'"
    assert network_config["optimizer"] in ["Adam", "SGD", "RMSprop", "Adagrad"], "optimizer must be one of 'Adam', 'SGD', 'RMSprop', or 'Adagrad'"
    assert network_config["lr"] > 0, "learning_rate must be greater than 0"

def setup_actor() -> MCTSAgent:
    actor_config, game_config, network_config = parse_config()
    game_implementation = Hex(**game_config)
    game = Game(game_implementation)
    actor = MCTSAgent(game=game, **actor_config)
    actor.setup_model(**network_config)
    return actor


def start() -> None:
    agent = setup_actor()
    print(f"Starting training.\n")
    agent.train()
    print("Playing against the network.")
    agent.play_against_network()
    print("Starting TOPP.")
    agent.play_topp()


if __name__ == '__main__':
    start()
