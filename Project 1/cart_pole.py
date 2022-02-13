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
        self.n_states = None
        self.current_state_parameters = self.current_state = None
        self.lower_bounds = [-4.8, -0.5, -0.41887903, -0.8726646259971648]
        self.upper_bounds = [4.8, 0.5, 0.41887903, 0.8726646259971648]
        # Keep track of angles for best episode
        self.angles = []
        self.initialize()

    def initialize(self) -> None:
        """This is the docstring for this function
        """
        self.generate_all_states()
        # Now generate all the permutations of the encoding of every state
        # self.states = np.array([self.decode_state(enc_state) for enc_state in self.state_encodings])
        # Actions: 0 corresponds to F = 10, 1 corresponds to F = -10
        self.actions = [0, 1]
        # Now create a state-to-encoded-state map that will be used for the neural network
        # Note: index = integer = current state, output = the encoding of that specific state
        # self.state_to_encoding = {
        #     self.states[i]: self.state_encodings[i] for i in range(self.n_states)}
    
    def get_legal_actions(self, state: int):
        return self.actions

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
        current_state = self.discretize_state(self.current_state_parameters)
        self.current_state = self.encoding_to_state[current_state]
        # self.current_state = self.decode_state(current_encoded_state)
        
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
        enc_next_state = self.discretize_state(next_state_parameters)
        next_state = self.encoding_to_state[enc_next_state]
        if abs(self.x) > 2.4 or abs(self.theta) > 0.21:
            done = True
            reward = -1
        elif self.elapsed_time >= 300:
            done = True
            reward = 10
        else:
            done = False
            reward = 1 #abs(self.current_state_parameters[2] - next_state_parameters[2])
        self.current_state_parameters = next_state_parameters
        self.current_state = next_state
        return self.current_state, reward, done
    
    def encode_state(self, state: int) -> tuple:
        encoded_state_tuple = self.state_to_encoding[state]
        encoded_state = tuple(encoded_state_tuple)
        return encoded_state
    
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
        all_encoded_states = np.array(list(product(*(range(0, self.buckets[i]) for i in range(len(self.buckets))))))
        self.n_states = len(all_encoded_states)
        self.states = []
        for state_num in range(self.n_states):
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
        # self.current_state = self.decode_state(encoding)


def test_something() -> None:
    game = CartPoleGame()
    test_state = [-0.03117389, 0.024164418, 0.0106699085, 0.04865943]
    game.set_state_manually(test_state)
    print(game.current_state_parameters, game.current_state)
    print(game.encode_state(game.current_state))
    test_action = 0
    game.step(test_action)
    next_parameters = game.current_state_parameters
    print(next_parameters)
    print(game.states)


def asdas():
    game = CartPoleGame()
    test_state = [10,10,10,10]
    game.set_state_manually(test_state)
    print(game.current_state)
    print(game.discretize_state(test_state))

if __name__ == "__main__":
    test_something()
