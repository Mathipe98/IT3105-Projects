from typing import Any, Dict, Union, List
import numpy as np
from itertools import product
from pprint import pprint

from tile_coding import create_tilings, get_tile_coding
from cart_pole import CartPoleGame

np.random.seed(123)


class Simworld:

    def __init__(self,
                 game: int,
                 n_episodes: int,
                 n_steps: int,
                 funcapp: bool,
                 network_dim: Union[tuple, None],
                 act_crit_params: List[float],
                 # act_lr: float,
                 # act_decay: float,
                 # act_disc: float,
                 # crit_lr: float,
                 # crit_decay: float,
                 # crit_disc: float,
                 epsilon: float,
                 display: Any,  # <- wtf does this do?
                 frame_delay: float,
                 **kwargs: Any) -> None:
        assert 0 <= game <= 2, "Game number must be between 0-2"
        assert n_episodes >= 1, "Number of episodes must be >= 1"
        assert n_steps > 0, "Number of steps must be > 0"
        if network_dim is not None:
            for dim in network_dim:
                assert dim > 0, "Network dimensions cannot be negative"
        for param in act_crit_params:
            assert 0.0 <= param <= 1, "Learning rate, decay, and discount parameters must be between 0-1"
        assert 0.0 <= epsilon <= 1, "Epsilon must be in the range 0-1"
        assert frame_delay >= 0.0, "Frame delay must be a positive value"
        self.game = game
        self.n_episodes = n_episodes
        self.n_steps = n_steps
        self.funcapp = funcapp
        self.network_dim = network_dim
        self.act_crit_params = act_crit_params
        self.epsilon = epsilon
        self.display = display
        self.frame_delay = frame_delay
        self.states = None
        self.tilings = None
        self.__dict__.update(kwargs)

        """
        Thoughts:
            - Create methods for generating (somehow) the general state of the game, some way to represent it
            - Create methods for getting a successor (child) state given a certain action
                * This because the actor will map states to actions, and therefore needs to consult the
                    simworld in order to know what state an action can lead to
            - Create method for receiving an immediate reinforcement (reward) of a state
                * This implies that a mapping must be constructed such that all states have some reinforcement
                * Was going to write something else here, but forgot it
            - Create method for generating the proper start and goal states
            - Create method for checking if a certain state is a goal state
        
        In general, try to come up with a way of representing the different games in similar fashions, such
        that all representations can use the same functionality.
        Maybe coarse coding for all games is possible, look into this.
        """

    def generate_states(self) -> np.ndarray:
        if self.game == 0:
            game = CartPoleGame()
            self.tilings = game.generate_encoding()
            self.states = generate_state_permutations(game.n_tilings, game.feat_bins)
        else:
            pass

    def get_states(self) -> np.ndarray:
        return self.states


def generate_state_permutations(n_tilings: int, feat_bins: List) -> np.ndarray:
    """This function generates all the possible states for a given problem.
    The states are solely based on the number of tiles, and the number of feature
    bins for each feature. The function creates all permutations for the potential
    binnings for each tile for each variable.
    The total number of states will be equal to:
    n_states = pow(prod(all bin values), n_tilings)

    Note: becomes slow for extremely large values. Less tilings decreases the runtime
    exponentially, so if the system becomes slow, this is the variable to reduce.

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


if __name__ == '__main__':
    test_object = Simworld(0, 1, 1, False, None, [0.1], 0, 0, 0)

    test_object.generate_states()
    print(len(test_object.get_states()))
    print(test_object.get_states().shape)
