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
                 l: float = 0.5, tau: float = 0.02, buckets: List = None) -> None:
        # Set variables for this particular game
        if buckets is None:
            buckets = (4, 4, 16, 8)
        self.g, self.m_c, self.m_p, self.l, self.tau = g, m_c, m_p, l, tau
        self.buckets = buckets
        # Create vectors for positions and velocities (and their derivatives) with length equal to the total number of steps
        self.T = 300
        self.elapsed_time = 0
        self.x = self.dx = self.d2x = None
        self.theta = self.dtheta = self.d2theta = None
        # Keep track of state
        self.encoding_to_state = {}
        self.state_to_encoding = {}
        self.states = self.actions = None
        # Keep track of the shape of the encoded state for use in the NN critic
        self.enc_shape = (len(buckets),)
        self.current_state_parameters = self.current_state = None
        #self.lower_bounds = [-4.8, -0.5, -0.41887903, -0.8726646259971648]
        #self.upper_bounds = [4.8, 0.5, 0.41887903, 0.8726646259971648]
        self.lower_bounds = [-2.4, -0.5, -0.21, -0.8726646259971648]
        self.upper_bounds = [2.4, 0.5, 0.21, 0.8726646259971648]
        # Keep track of angles for best episode
        self.angles = []
        self.initialize()

    def initialize(self) -> None:
        """This is the docstring for this function
        """
        self.generate_all_states()
        self.actions = [0, 1]

    def get_legal_actions(self, state: int):
        return self.actions

    def reset(self) -> int:
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
        current_state = self.discretize_state(self.current_state_parameters)
        self.current_state = self.encoding_to_state[current_state]
        return self.current_state

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

        p1 = self.l * (4/3 - (self.m_p * costheta ** 2) / total_mass)
        p2 = costheta * (-force - self.m_p * self.l * dtheta ** 2 * sintheta)
        d2theta = (self.g * sintheta + p2) / p1

        p3 = self.m_p * self.l * (dtheta ** 2 * sintheta - d2theta * costheta)
        d2x = (force + p3) / total_mass
        self.dtheta = dtheta + self.tau * d2theta
        self.dx = dx + self.tau * d2x
        self.theta = theta + self.tau * dtheta
        self.x = x + self.tau * dx
        next_state_parameters = [
            self.x,
            self.dx,
            self.theta,
            self.dtheta
        ]
        enc_next_state = self.discretize_state(next_state_parameters)
        next_state = self.encoding_to_state[enc_next_state]
        if abs(self.x) > 2.4 or abs(self.theta) > 0.21:
            done = True
            reward = -100
        elif self.elapsed_time >= 300:
            done = True
            reward = 1
        else:
            done = False
            reward = 0
        self.current_state_parameters = next_state_parameters
        self.current_state = next_state
        return self.current_state, reward, done

    def encode_state(self, state: int) -> np.ndarray:
        """Method that takes in a state number, and encodes this state in a general way
        such that a neural network can train and predict on it.
        The reason that this method returns an array, while the "states" attribute of the
        class uses tuples, is because tuples are hashable, while tuples are not.
        However this method will only be called within the neural network critic, so it
        won't affect anything else.

        Args:
            state (int): State number representation of the state in question

        Returns:
            np.ndarray: Result array of aforementioned encoding
        """
        encoded_state_tuple = self.state_to_encoding[state]
        return np.array(encoded_state_tuple)

    def decode_state(self, enc_state: np.ndarray) -> int:
        decoded_state = \
            enc_state[0] * self.buckets[1] * self.buckets[2] * self.buckets[3] +\
            enc_state[1] * self.buckets[2] * self.buckets[3] +\
            enc_state[2] * self.buckets[3] +\
            enc_state[3]
        return decoded_state

    def discretize_state(self, state_parameters: List) -> tuple:
        """
        Takes an observation of the environment and aliases it.
        By doing this, very similar observations can be treated
        as the same and it reduces the state space so that the 
        Q-table can be smaller and more easily filled.

        Input:
        state_parameters (List): Tuple containing 4 floats describing the current
                     state of the environment.

        Output:
        discretized (tuple): Tuple containing 4 non-negative integers smaller 
                             than n where n is the number in the same position
                             in the buckets list.
        """
        discretized = list()
        for i in range(len(state_parameters)):
            scaling = ((state_parameters[i] + abs(self.lower_bounds[i]))
                       / (self.upper_bounds[i] - self.lower_bounds[i]))
            new_obs = int(round((self.buckets[i] - 1) * scaling))
            new_obs = min(self.buckets[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)

    def generate_all_states(self) -> None:
        all_encoded_states = np.array(
            list(product(*(range(0, self.buckets[i]) for i in range(len(self.buckets))))))
        n_states = len(all_encoded_states)
        self.states = []
        for state_num in range(n_states):
            encoded_state = tuple(all_encoded_states[state_num])
            self.state_to_encoding[state_num] = encoded_state
            self.encoding_to_state[encoded_state] = state_num
            self.states.append(state_num)

    def set_state_manually(self, parameters: List) -> None:
        self.reset()
        self.x = parameters[0]
        self.dx = parameters[1]
        self.theta = parameters[2]
        self.dtheta = parameters[3]
        self.current_state_parameters = [
            self.x, self.dx, self.theta, self.dtheta]
        encoding = self.discretize_state(self.current_state_parameters)
        self.current_state = self.encoding_to_state[encoding]
