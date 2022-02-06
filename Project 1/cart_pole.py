from typing import Any, Dict, Union, List, Tuple
import numpy as np
import random
from itertools import product
from pprint import pprint

from tile_coding import create_tilings, get_tile_coding


class CartPoleGame():
    """Docstring here
    """

    def __init__(self, g: float = 9.81, m_c: float = 1, m_p: float = 1,
                 l: float = 0.5, tau: float = 0.02, n_tilings: int = 1,
                 feat_bins: List = None) -> None:
        # super().__init__(**global_config)
        # Set variables for this particular game
        if feat_bins is None:
            feat_bins = [4, 4, 8, 8, 2]
        self.g, self.m_c, self.m_p, self.l, self.tau = g, m_c, m_p, l, tau
        self.n_tilings, self.feat_bins = n_tilings, feat_bins
        # State stuff
        self.tilings = self.states = self.actions = None
        self.state_to_encoding = self.encoding_to_state = None
        self.transition_normal = 1
        self.transition_lose = -10
        self.transition_win = 10
        self.n_states = self.n_actions = None
        # Create vectors for positions and velocities (and their derivatives) with length equal to the total number of steps
        self.T = 300
        self.elapsed_time = 0
        self.x = self.dx = self.d2x = None
        self.theta = self.dtheta = self.d2theta = None
        # Keep track of state
        self.current_state_parameters = self.current_state = None
        self.initialize()

    def initialize(self) -> None:
        """This is the docstring for this function
        """
        # The tilings are the arrays we use to encode a particular state
        self.tilings = self.generate_encoding_tiles()
        # Now generate all the permutations of the encoding of every state
        state_encodings = self.generate_all_encoded_states(
            self.n_tilings, self.feat_bins)
        # Now make an array containing the actual states (integers)
        self.states = np.array(range(0, len(state_encodings)))
        # Actions: 0 corresponds to F = 10, 1 corresponds to F = -10
        self.actions = [0, 1]
        self.n_states, self.n_actions = len(self.states), len(self.actions)
        # Now create a state-to-encoded-state map that will be used for the neural network
        # Note: index = integer = current state, output = the encoding of that specific state
        self.state_to_encoding = {
            self.states[i]: state_encodings[i] for i in range(len(self.states))}
        # Create another map that we'll use to extract the state from the encoded state
        self.encoding_to_state = {}
        for l in range(len(self.states)):
            state = l
            encoded_state = self.state_to_encoding[state]
            encoded_state_string = str(encoded_state)
            self.encoding_to_state[encoded_state_string] = state

    def start_episode(self) -> None:
        """Method for starting a new episode, which includes resetting
        all states and state parameters in the simworld
        """
        # Reset the amount of time that has passed and reset the arrays
        self.elapsed_time = 0
        steps = int(np.ceil(self.T / self.tau))
        self.x = np.zeros(steps)
        self.dx = np.zeros(steps)
        self.d2x = np.zeros(steps)
        self.theta = np.zeros(steps)
        self.dtheta = np.zeros(steps)
        self.d2theta = np.zeros(steps)
        # Randomly assign the first theta-value
        self.theta[0] = np.random.uniform(-0.21, 0.21)
        # Set current state (and parameters) to start
        self.current_state_parameters = [
            self.x[0], self.dx[0], self.theta[0], self.dtheta[0], self.elapsed_time]
        current_encoded_state = get_tile_coding(
            self.current_state_parameters, self.tilings)
        self.current_state = self.encoding_to_state[str(current_encoded_state)]
        # Create a dictionary that acts as a cache for intermediate states
        # (transition between generating a child state, and setting the state to the child state)
        self.cache = {}

    def generate_child_state(self, action: int) -> Tuple[int, int]:
        """This method calculates and returns the next state given the current state.
        Since this simworld keeps track of its own state parameters, we only need to know
        in order to calculate the next state's parameters, its encoding, and thereby the
        next state itself.


        Args:
            action (int): Integer, 0 or 1, which corresponds to applying a force in a direction
                            0 => 10, 1 => -10
        Returns:
            Tuple[int, int]: Tuple containing 1. the next state, and 2. the reward of the transition
        """
        action = 10 - 20 * action
        elapsed_time = self.current_state_parameters[-1]
        t = round(elapsed_time / self.tau) + 1
        self.theta[t] = self.theta[t-1] + self.tau * self.dtheta[t-1]
        self.dtheta[t] = self.dtheta[t-1] + self.tau * self.d2theta[t-1]
        self.d2theta[t] = self.compute_d2theta(action, t)
        self.x[t] = self.x[t-1] + self.tau * self.dx[t-1]
        self.dx[t] = self.dx[t-1] + self.tau * self.d2x[t-1]
        self.d2x[t] = self.compute_d2x(action, t)
        self.elapsed_time = round(self.elapsed_time + self.tau, 6)
        next_state_parameters = [
            self.x[t],
            self.dx[t],
            self.theta[t],
            self.dtheta[t],
            self.elapsed_time
        ]
        encoded_next_state = get_tile_coding(
            next_state_parameters, self.tilings)
        next_state = self.encoding_to_state[str(encoded_next_state)]
        reward = self.transition_win if self.is_winning_state(next_state) \
            else self.transition_lose if self.is_losing_state(next_state) \
            else self.transition_normal
        self.cache["next_params"] = next_state_parameters
        self.cache["next_state"] = next_state
        return self.current_state, reward
    
    def set_child_state(self) -> None:
        self.current_state = self.cache["next_state"]
        self.current_state_parameters = self.cache["next_params"]

    def compute_d2x(self, F: float, t: int) -> float:
        """Method for calculating the second derivative of the angular velocity

        Args:
            F (float): Force applied
            t (int): Current time in the simworld

        Returns:
            float: Result of aforementioned calculations
        """
        var_1 = self.m_p * self.l * \
            (self.dtheta[t]**2 * np.sin(self.theta[t]) -
             self.d2theta[t] * np.cos(self.theta[t]))
        var_2 = self.m_c + self.m_p
        result = (F + var_1) / var_2
        return result

    def compute_d2theta(self, F: float, t: int) -> float:
        """Blabla

        Args:
            F (float): [description]
            t (int): [description]

        Returns:
            float: [description]
        """
        denom_part = (self.m_p * self.theta[t] ** 2) / (self.m_c + self.m_p)
        denominator = self.l * (4/3 - denom_part)
        numer_part = (-F - self.m_p * self.l *
                      self.dtheta[t]**2 * np.sin(self.theta[t])) / (self.m_c + self.m_p)
        numerator = (
            self.g * np.sin(self.theta[t]) + np.cos(self.theta[t]) * numer_part)
        result = numerator / denominator
        return result

    def is_losing_state(self, state: int) -> bool:
        """Checks whether a state is a losing state. It does this by
        extracting the cart velocity and the pole angular velocity, and checking
        if the encoding of the state results in bins for the aforementioned
        velocities that are at the outer points.
        Basically: if x falls in bin 0, this means that x < -2.4
        If falls into bin x_limit - 1, which most often will be 3, this means that x > 2.4
        The same logic applies to the angle, but with different bins.
        Thus we can determine if a state is losing just by its encoding, and therefore
        the agent needs not keep track of any state parameters.

        Args:
            state (int): The state to be tested

        Returns:
            bool: Classification of the state (losing/not losing)
        """
        encoded_state = self.state_to_encoding[state]
        x_pos = encoded_state[0][0]
        theta_pos = encoded_state[0][2]
        x_limit = self.feat_bins[0] - 1
        theta_limit = self.feat_bins[2] - 1
        # print(f"Limits: {(x_limit, theta_limit)}")
        return x_pos == 0 or x_pos == x_limit or \
            theta_pos == 0 or theta_pos == theta_limit

    def is_winning_state(self, state: int) -> bool:
        """Checks whether all the values for the encoded state's time variable
        fall into bin 1. It basically means that if the last column of the encoded
        state consists of all 1's, then t >= 300 and the state is winning.

        Args:
            state (int): The state to be tested

        Returns:
            bool: Classification of the state (winning/not winning)
        """
        encoded_state = self.state_to_encoding[state]
        return np.all((encoded_state[:, [-1]] == 1))

    def generate_encoding_tiles(self) -> np.ndarray:
        """This method generates the tile-encoding for the cart problem.

        Returns:
            np.ndarray: Array that is used for indexing and encoding the features
                        in the problem (i.e. positions and velocities + time)
        """
        x_range = [-2.4, 2.4]
        x_v_range = [-1, 1]
        ang_range = [-0.21, 0.21]
        ang_v_range = [-0.1, 0.1]
        t_range = [0, 300]
        ranges = [x_range, x_v_range, ang_range, ang_v_range, t_range]
        # Create a nested list that uses the same bins for every tiling
        bins = [bin for bin in self.feat_bins]
        bins = [bins[:] for _ in range(self.n_tilings)]
        offset = 0
        offset_list = []
        for _ in range(self.n_tilings):
            current = []
            for i in range(len(ranges) - 1):
                a = ranges[i][0]
                b = ranges[i][1]
                ab_sum = abs(a) + abs(b)
                # Let the offset for a particular feature be 20% of the feature itself
                # Then double that every iteration with offset variable
                feat_offset = round(ab_sum * 0.2 * offset, 4)
                current.append(feat_offset)
            # Append 150 last because time offset is constant
            current.append(150)
            offset_list.append(current)
            offset += 1
        tilings = create_tilings(ranges, self.n_tilings, bins, offset_list)
        return tilings

    def generate_all_encoded_states(self, n_tilings: int, feat_bins: List) -> np.ndarray:
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


if __name__ == "__main__":
    test = CartPoleGame()
    test.initialize()
    test_state = 50
    enc_state = test.state_to_encoding[test_state]
    print(enc_state)
    print(test.is_winning_state(test_state))
