from typing import Any, Dict, Union
import numpy as np
from tile_coding import create_tilings, get_tile_coding


class Simworld:

    def __init__(self,
                 game: int,
                 n_episodes: int,
                 n_steps: int,
                 funcapp: bool,
                 network_dim: Union[tuple, None],
                 act_crit_params: list[float],
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

    def generate_states(self, game: int) -> np.ndarray:
        if game == 0:
            # Game is pole-balancing, we need a general way to represent the state
            # State parameters: cart position, cart speed, pole angle, pole angular velocity (x, x', theta, theta')
            # Thoughts: binning; for each variable, put the values into discrete bins (buckets) of values
            # Include tau (timestep) in the state, but only 2 bins; 1 bin is when tau < 300 (which is the goal), and the other is tau >= 300. Then we can check whether tau is in the second bin when checking if winning state
            # First create ranges for all variables such that we can check states easily in bins
            x_range = [-2.4, 2.4] # Note: when the state for this is 0 or the length of the bin, then we lose. Same goes for angle
            x_v_range = [-1, 1]
            ang_range = [-0.21, 0.21]
            ang_v_range = [-0.1, 0.1]
            t_range = [0, 300]
            ranges = [x_range, x_v_range, ang_range, ang_v_range, t_range]
            # Setup bin (bucket) parameters
            n_tilings = 3
            x_bins = 4
            x_v_bins = 4
            ang_bins = 8
            ang_v_bins = 8
            t_bins = 2
            # Create a nested list that uses the same bins for every tiling
            bins = [x_bins, x_v_bins, ang_bins, ang_v_bins, t_bins]
            bins = [bins[:] for _ in range(n_tilings)]
            
            pass
        else:
            pass
    
 
