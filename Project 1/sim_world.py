from typing import Any, Dict, Union, List, Tuple
import numpy as np
import random
from itertools import product
from pprint import pprint

from tile_coding import create_tilings, get_state_encoding

# np.random.seed(123)


class SimWorld:

    def __init__(self, game_number: int, n_episodes: int, maximum_steps: int,
                 td_n_steps: int, funcapp: bool, network_dim: Union[tuple, None],
                 act_crit_params: List[float],
                 # act_lr: float, act_decay: float, act_disc: float,
                 # crit_lr: float, crit_decay: float, crit_disc: float,
                 epsilon: float, display: Any, frame_delay: float) -> None:
        self.game_number = game_number
        self.n_episodes = n_episodes
        self.maximum_steps = maximum_steps
        self.td_n_steps = td_n_steps
        self.funcapp = funcapp
        self.network_dim = network_dim
        self.act_crit_params = act_crit_params
        self.epsilon = epsilon
        self.display = display
        self.frame_delay = frame_delay

    # def initialize(self, game_params: Dict) -> None:
    #     # Cart pole
    #     if self.game_number == 1:
    #         self.game = CartPoleGame(**game_params)
    #         self.tilings = self.game.generate_encoding()
    #         self.states = self.generate_all_states(self.game.n_tilings, self.game.feat_bins)
    #         self.actions = [10, -10]
    #         # Get 1 point every timestep for not falling down
    #         self.transition_normal = 1
    #         # Get -1000 points for losing
    #         self.transition_lose = -1000
    #         # Get 1000 points for winning
    #         self.transition_win = 1000
    #     # Towers of hanoi
    #     elif self.game_number == 2:
    #         self.game = None
    #         self.tilings = self.game.generate_encoding()
    #         self.states = self.generate_all_states(self.game.n_tilings, self.game.feat_bins)
    #         # Actions: each number corresponds to the peg on which to place a disc
    #         # Note to self: maybe make large negative reward for placing disc on same peg (or maybe just negative stationary transition reward so it learns that spending a lot of time is bad)
    #         self.actions = [0, 1, 2]
    #         # Get 1 point every timestep for not falling down
    #         # self.transition_normal = 1
    #         # Get -1000 points for losing
    #         # self.transition_lose = -1000
    #         # Get 1000 points for winning
    #        #  self.transition_win = 1000
    #     elif self.game_number == 3:
    #         self.game = None
    #         self.tilings = self.game.generate_encoding()
    #         self.states = self.generate_all_states(self.game.n_tilings, self.game.feat_bins)
    #         # Actions: each number corresponds to whether or not to bet (1 to bet, 0 to not bet)
    #         self.actions = [0, 1]
    #         # Get 1 point every timestep for not falling down
    #         # self.transition_normal = 1
    #         # Get -1000 points for losing
    #         # self.transition_lose = -1000
    #         # Get 1000 points for winning
    #        #  self.transition_win = 1000

    
    # def new_episode(self) -> None:
    #     if self.game_number == 0:
    #         self.game = CartPoleGame()

    # def get_next_state(self, features: List, action: float) -> Tuple[List, np.ndarray]:
    #     next_feature_values = self.game.get_next_state(
    #         features, action)
    #     next_state = get_tile_coding(next_feature_values, self.tilings)
    #     return next_feature_values, next_state

    # def is_winning_state(self, state: np.ndarray) -> bool:
    #     return self.game.is_winning_state(state)

    # def is_losing_state(self, state: np.ndarray) -> bool:
    #     return self.game.is_losing_state(state)

