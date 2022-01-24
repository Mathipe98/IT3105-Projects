from typing import Any, Dict, Union, List, Tuple
import numpy as np
import random
from itertools import product
from pprint import pprint

from tile_coding import create_tilings, get_tile_coding
from cart_pole import CartPoleGame

np.random.seed(123)


class Simworld:

    def __init__(self, game_number: int, n_episodes: int, n_steps: int,
                 funcapp: bool, network_dim: Union[tuple, None],
                 act_crit_params: List[float],
                 # act_lr: float, act_decay: float, act_disc: float,
                 # crit_lr: float, crit_decay: float, crit_disc: float,
                 epsilon: float, display: Any, frame_delay: float) -> None:
        self.game_number = game_number
        self.n_episodes = n_episodes
        self.n_steps = n_steps
        self.funcapp = funcapp
        self.network_dim = network_dim
        self.act_crit_params = act_crit_params
        self.epsilon = epsilon
        self.display = display
        self.frame_delay = frame_delay
        self.tilings = None
        self.game = None
        self.states = None
        self.initialize()

    def initialize(self) -> None:
        if self.game_number == 0:
            self.game = CartPoleGame()
            self.tilings = self.game.generate_encoding()
            # self.states = generate_state_permutations(
            # game.n_tilings, game.feat_bins)
    
    def new_episode(self) -> None:
        if self.game_number == 0:
            self.game = CartPoleGame()

    def get_next_state(self, features: List, action: float) -> Tuple[List, np.ndarray]:
        next_feature_values = self.game.get_next_state_parameters(
            features, action)
        next_state = get_tile_coding(next_feature_values, self.tilings)
        return next_feature_values, next_state

    def is_winning_state(self, state: np.ndarray) -> bool:
        return self.game.is_winning_state(state)

    def is_losing_state(self, state: np.ndarray) -> bool:
        return self.game.is_losing_state(state)


def generate_state_permutations(n_tilings: int, feat_bins: List) -> np.ndarray:
    """This function generates all the possible states for a given problem.
    The states are solely based on the number of tiles, and the number of feature
    bins for each feature.
    Args:
        feat_bins (List): List containing the number of bins for each variable

    Returns:
        np.ndarray: Array of every possible state in the cart pole world
    """
    # Create a result array that will contain all the possible states
    all_possible_states = []
    # Create a list of ranges for creating cross products of every value
    range_inputs = []
    for bin in feat_bins:
        range_inputs.append(range(0, bin))
    # Now permute all possible values to create all possible states
    permutations = []
    for _ in range(n_tilings):
        permutations.append(list(product(*range_inputs)))
    cross_product = product(*permutations)
    for v in cross_product:
        all_possible_states.append([list(perm) for perm in v])
    return np.array(all_possible_states, dtype='object')


def test_random_survival() -> None:
    test_features = [-2.3, -0.1, -0.20, -0.1, 0]
    world = Simworld(0, 1, 1, False, None, [0.1], 0, 0, 0)
    actions = [-10, 10]
    features = test_features
    state = None
    losing = False
    while not losing:
        action = random.choice(actions)

        features, state = world.get_next_state(features, action)
        losing = world.is_losing_state(state)
        print(f"Features: {features}")
        print(f"State: {state}")
        print(f"Is winning: {world.is_winning_state(state)}")
        print(f"Is losing: {losing}\n")

if __name__ == '__main__':
    test_object = Simworld(0, 1, 1, False, None, [0.1], 0, 0, 0)
    print(test_object.tilings)
    test_random_survival()
