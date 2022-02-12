from typing import Any, Dict, Union, List, Tuple
import numpy as np
import random
from itertools import product
from pprint import pprint

from tile_coding import create_tilings, get_state_encoding


class CartPoleGame():
    """Docstring here
    """

    def __init__(self, g: float = 9.8, m_c: float = 1, m_p: float = 0.1,
                 l: float = 0.5, tau: float = 0.02, n_tilings: int = 1,
                 feat_bins: List = None) -> None:
        # super().__init__(**global_config)
        # Set variables for this particular game
        if feat_bins is None:
            feat_bins = [4, 4, 16, 8]
        self.g, self.m_c, self.m_p, self.l, self.tau = g, m_c, m_p, l, tau
        self.n_tilings, self.feat_bins = n_tilings, feat_bins
        # State stuff
        self.tilings = self.states = self.state_encodings = self.actions = None
        self.state_to_encoding = self.encoding_to_state = None
        self.n_states = self.n_actions = None
        # Create vectors for positions and velocities (and their derivatives) with length equal to the total number of steps
        self.T = 300
        self.elapsed_time = 0
        self.x = self.dx = self.d2x = None
        self.theta = self.dtheta = self.d2theta = None
        # Keep track of state
        self.current_state_parameters = self.current_state = None
        self.lower_bounds = [-4.8, -0.5, -0.41887903, -0.8726646259971648]
        self.upper_bounds = [4.8, 0.5, 0.41887903, 0.8726646259971648]
        # Keep track of angles for best episode
        self.angles = []
        self.initialize()

    def initialize(self) -> None:
        """This is the docstring for this function
        """
        # The tilings are the arrays we use to encode a particular state
        self.tilings = self.generate_encoding_tiles()
        # Now generate all the permutations of the encoding of every state
        self.states = np.array([self.decode_state(enc_state) for enc_state in self.state_encodings])
        # Actions: 0 corresponds to F = 10, 1 corresponds to F = -10
        self.actions = [0, 1]
        self.n_states, self.n_actions = len(self.states), len(self.actions)
        # Now create a state-to-encoded-state map that will be used for the neural network
        # Note: index = integer = current state, output = the encoding of that specific state
        self.state_to_encoding = {
            self.states[i]: self.state_encodings[i] for i in range(self.n_states)}

    def reset(self) -> None:
        """Method for starting a new episode, which includes resetting
        all states and state parameters in the simworld
        """
        # Reset the amount of time that has passed and reset the arrays
        self.elapsed_time = 0
        self.x = 0
        self.dx = 0
        self.d2x = 0
        self.dtheta = 0
        self.d2theta = 0
        # Randomly assign the first theta-value
        self.theta = np.random.uniform(-0.21, 0.21)
        # Set current state (and parameters) to start
        self.current_state_parameters = [
            self.x, self.dx, self.theta, self.dtheta]
        current_encoded_state = get_state_encoding(
            self.current_state_parameters, self.tilings)
        self.current_state = self.decode_state(current_encoded_state)
        # Create a dictionary that acts as a cache for intermediate states
        # (transition between generating a child state, and setting the state to the child state)
        self.cache = {}
    
    def set_state_manually(self, parameters: List) -> None:
        self.reset()
        self.x = parameters[0]
        self.dx = parameters[1]
        self.theta = parameters[2]
        self.dtheta = parameters[3]
        self.current_state_parameters = [
            self.x, self.dx, self.theta, self.dtheta]
        current_encoded_state = get_state_encoding(
            self.current_state_parameters, self.tilings)
        self.current_state = self.decode_state(current_encoded_state)
        


    def step(self, action: int) -> Tuple[int, int, bool]:
        """This method calculates and returns the next state given the current state.
        Since this simworld keeps track of its own state parameters, we only need to know
        in order to calculate the next state's parameters, its encoding, and thereby the
        next state itself.


        Args:
            action (int): Integer, 0 or 1, which corresponds to applying a force in a direction
                            0 => -10, 1 => 10
        Returns:
            Tuple[int, int, bool]: Tuple containing 1. the next state, 2. the reward of the transition,
                                and 3. whether or not the resulting state is an end state
        """
        force = -10 + 20 * action
        self.elapsed_time += self.tau
        total_mass = self.m_p + self.m_c
        x, dx, theta, dtheta = self.current_state_parameters
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (
            force + self.m_p * self.l * dtheta ** 2 * sintheta
        ) / total_mass
        thetaacc = (self.g * sintheta - costheta * temp) / (
            self.l * (4.0 / 3.0 - self.m_p * costheta ** 2 / total_mass)
        )
        xacc = temp - self.m_p * self.l * thetaacc * costheta / total_mass
        self.x = x + self.tau * dx
        self.dx = dx + self.tau * xacc
        self.theta = theta + self.tau * dtheta
        self.dtheta = dtheta + self.tau * thetaacc
        next_state_parameters = [
            self.x,
            self.dx,
            self.theta,
            self.dtheta
        ]
        encoded_next_state = get_state_encoding(
            next_state_parameters, self.tilings)
        next_state = self.decode_state(encoded_next_state)
        if abs(self.x) > 2.4 or abs(self.theta) > 0.21:
            done = True
            reward = -1
        elif self.elapsed_time >= 300:
            done = True
            reward = 2
        else:
            done = False
            reward = 0 #abs(self.current_state_parameters[2] - next_state_parameters[2])
        self.current_state_parameters = next_state_parameters
        self.current_state = next_state
        return self.current_state, reward, done
    
    def decode_state(self, enc_state: np.ndarray) -> int:
        decoded_state = \
            enc_state[0] * self.feat_bins[1] * self.feat_bins[2] * self.feat_bins[3] +\
            enc_state[1] * self.feat_bins[2] * self.feat_bins[3] +\
            enc_state[2] * self.feat_bins[3] +\
            enc_state[3]
        return decoded_state

    def compute_d2x(self, F: float, t: int) -> float:
        """Method for calculating the second derivative of the angular velocity

        Args:
            F (float): Force applied
            t (int): Current time in the simworld

        Returns:
            float: Result of aforementioned calculations
        """
        theta = self.theta[t-1]
        dtheta = self.dtheta[t-1]
        d2theta = self.d2theta[t-1]
        var_1 = self.m_p * self.l * (dtheta**2 * np.sin(theta) - d2theta * np.cos(theta))
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
        theta = self.theta[t-1]
        dtheta = self.dtheta[t-1]
        denom_part = (self.m_p * np.cos(theta) ** 2) / (self.m_c + self.m_p)
        denominator = self.l * (4/3 - denom_part)
        numer_part = (-F - self.m_p * self.l *
                      dtheta**2 * np.sin(theta)) / (self.m_c + self.m_p)
        numerator = (
            self.g * np.sin(theta) + np.cos(theta) * numer_part)
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
        # encoded_state = self.state_to_encoding[state]
        # return np.all((encoded_state[:, [-1]] == 1))
        return self.elapsed_time >= 300

    def discretize_state(self, obs):
        """
        Takes an observation of the environment and aliases it.
        By doing this, very similar observations can be treated
        as the same and it reduces the state space so that the 
        Q-table can be smaller and more easily filled.
        
        Input:
        obs (tuple): Tuple containing 4 floats describing the current
                     state of the environment.
        
        Output:
        discretized (tuple): Tuple containing 4 non-negative integers smaller 
                             than n where n is the number in the same position
                             in the buckets list.
        """
        discretized = list()
        for i in range(len(obs)):
            scaling = ((obs[i] + abs(self.lower_bounds[i])) 
                       / (self.upper_bounds[i] - self.lower_bounds[i]))
            new_obs = int(round((self.feat_bins[i] - 1) * scaling))
            new_obs = min(self.feat_bins[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return list(discretized)
    
    def generate_all_states(self) -> None:
        

    def generate_encoding_tiles(self) -> np.ndarray:
        """This method generates the tile-encoding for the cart problem.

        Returns:
            np.ndarray: Array that is used for indexing and encoding the features
                        in the problem (i.e. positions and velocities + time)
        """
        x_range = [-2.4, 2.4]
        x_v_range = [-0.5, 0.5]
        ang_range = [-0.21, 0.21]
        ang_v_range = [-0.5, 0.5]
        ranges = [x_range, x_v_range, ang_range, ang_v_range]
        # Create a nested list that uses the same bins for every tiling
        bins = [bin for bin in self.feat_bins]
        bins = [bins[:] for _ in range(self.n_tilings)]
        offset = 0
        offset_list = []
        for _ in range(self.n_tilings):
            current = []
            for i in range(len(ranges)):
                a = ranges[i][0]
                b = ranges[i][1]
                ab_sum = abs(a) + abs(b)
                # Let the offset for a particular feature be 20% of the feature itself
                # Then double that every iteration with offset variable
                feat_offset = round(ab_sum * 0.2 * offset, 4)
                current.append(feat_offset)
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


def test_something() -> None:
    game = CartPoleGame()
    test_state = [-0.03117389, 0.024164418, 0.0106699085, 0.04865943]
    game.set_state_manually(test_state)
    print(game.current_state_parameters, game.current_state)
    print(game.d2theta)
    print(game.d2x)
    test_action = 0
    game.step(test_action)
    next_parameters = game.current_state_parameters
    print(next_parameters)


def asdas():
    game = CartPoleGame()
    test_state = [10,10,10,10]
    game.set_state_manually(test_state)
    print(game.current_state)
    print(game.discretize_state(test_state))
    print(get_state_encoding(test_state, game.tilings))

if __name__ == "__main__":
    asdas()
