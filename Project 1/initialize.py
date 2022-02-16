from typing import Dict, Tuple

from Agent import Agent
from hanoi import Hanoi
from cart_pole import CartPoleGame
from gambler import Gambler

import configparser
import ast
import os
import pathlib
os.chdir(pathlib.Path(__file__).parent.resolve())


def parse_config() -> Tuple[Dict, Dict, Dict, Dict, int]:
    """Function that will parse a local configuration file in order to easily setup
    different parameters for the agent and simworld(s).

    Returns:
        Dict: Dictionary containing all the necessary parameters
    """
    config = configparser.ConfigParser()
    config.read('./config.txt')
    global_config = {}
    cartpole_config = {}
    hanoi_config = {}
    gambler_config = {}
    for section in config.sections():
        if section == "GLOBALS":
            global_config["n_episodes"] = int(
                config.get(section, "n_episodes"))
            global_config["max_steps"] = int(config.get(section, "max_steps"))
            global_config["use_nn"] = ast.literal_eval(
                config.get(section, "use_nn"))
            global_config["network_dim"] = ast.literal_eval(
                config.get(section, "network_dim"))
            global_config["train_interval"] = int(
                config.get(section, "train_interval"))
            global_config["a_alpha"] = float(config.get(section, "a_alpha"))
            global_config["c_alpha"] = float(config.get(section, "c_alpha"))
            global_config["gamma"] = float(config.get(section, "gamma"))
            global_config["lamb"] = float(config.get(section, "lamb"))
            global_config["epsilon_start"] = float(
                config.get(section, "epsilon_start"))
            global_config["epsilon_finish"] = float(
                config.get(section, "epsilon_finish"))
            global_config["display"] = ast.literal_eval(
                config.get(section, "display"))
            global_config["frame_delay"] = float(
                config.get(section, "frame_delay"))
            global_config["chosen_game"] = int(config.get(section, "chosen_game"))
        elif section == "CARTPOLE":
            cartpole_config["l"] = float(config.get(section, "l"))
            cartpole_config["m_p"] = float(config.get(section, "m_p"))
            cartpole_config["g"] = float(config.get(section, "g"))
            cartpole_config["tau"] = float(config.get(section, "tau"))
        elif section == "HANOI":
            hanoi_config["n_pegs"] = int(config.get(section, "n_pegs"))
            hanoi_config["n_discs"] = int(config.get(section, "n_discs"))
        elif section == "GAMBLER":
            gambler_config["p_win"] = float(config.get(section, "p_win"))
    verify_configs(global_config, cartpole_config, hanoi_config, gambler_config)
    return global_config, cartpole_config, hanoi_config, gambler_config


def verify_configs(global_config: Dict, cartpole_config: Dict, hanoi_config: Dict, gambler_config: Dict) -> None:
    network_dim = global_config["network_dim"]
    assert network_dim[-1] == 1, "Last element in network dimensions must be equal to 1 due to implementation"
    epsilon1, epsilon2 = global_config["epsilon_start"], global_config["epsilon_finish"]
    assert 0 <= epsilon1 <= 1 and 0 <= epsilon2 <= 1, "Epsilon values must be between 0 and 1"
    l = cartpole_config["l"]
    assert 0.1 <= l <= 1, "Pole length must be between 0.1 and 1"
    m_p = cartpole_config["m_p"]
    assert 0.05 <= m_p <= 0.5, "Pole mass must be between 0.05 and 0.5"
    g = cartpole_config["g"]
    assert -15 <= g <= -5, "Gravity must be between -15 and -5"
    tau = cartpole_config["tau"]
    assert 0.01 <= tau <= 0.1, "Timestep (tau) must be between 0.01 and 0.1"
    n_pegs = hanoi_config["n_pegs"]
    assert 3 <= n_pegs <= 5, "Number of pegs must be between 3 and 5"
    n_discs = hanoi_config["n_discs"]
    assert 2 <= n_discs <= 6, "Number of discs must be between 2 and 6"
    p_win = gambler_config["p_win"]
    assert 0 <= p_win <= 1, "Probability of winning must be between 0 and 1"

def setup_agent() -> Agent:
    global_config, cartpole_config, hanoi_config, gambler_config = parse_config()
    chosen_game = global_config["chosen_game"]
    if chosen_game == 1:
        game = CartPoleGame(**cartpole_config)
    elif chosen_game == 2:
        game = Hanoi(**hanoi_config)
    else:
        game = Gambler(**gambler_config)
    agent = Agent(**global_config, game=game)
    return agent


def start() -> None:
    agent = setup_agent()
    agent.train()
    if agent.display:
        agent.visualize()


if __name__ == '__main__':
    start()
