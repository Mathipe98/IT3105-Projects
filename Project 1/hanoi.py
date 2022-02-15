from typing import Any, Dict, Union, List, Tuple
import numpy as np
import random
from itertools import product, permutations
from pprint import pprint
import copy
from tile_coding import create_tilings, get_state_encoding
import sys
sys.setrecursionlimit(1000000)


class Hanoi():
    """Docstring here
    """

    def __init__(self, n_pegs: int=3, n_discs: int=3):
        self.n_pegs = n_pegs
        self.n_discs = n_discs
        self.current_state_parameters = None
        self.current_state = None
        self.encoded_states = []
        self.states = []
        self.state_to_encoding = {}
        self.encoding_to_state = {}
        # Keep track of the shape of the encoded state for use in the NN critic
        self.enc_shape = (n_pegs,)
        self.counter = 0
        self.initialize()
    
    def initialize(self) -> None:
        self.current_state_parameters = []
        self.current_state_parameters.append(list(range(1,self.n_discs+1)))
        for _ in range(self.n_pegs - 1):
            self.current_state_parameters.append([])
        self.generate_all_states()
        assert len(self.encoded_states) != 0, "States not generated"
        for state_number in range(len(self.encoded_states)):
            encoded_state_tuple = tuple(tuple(x) for x in self.encoded_states[state_number])
            self.state_to_encoding[state_number] = encoded_state_tuple
            self.encoding_to_state[encoded_state_tuple] = state_number
            self.states.append(state_number)
        key = tuple(tuple(x) for x in self.current_state_parameters)
        self.current_state = self.encoding_to_state[key]

    def reset(self) -> int:
        self.current_state_parameters = []
        self.current_state_parameters.append(list(range(1,self.n_discs+1)))
        for _ in range(self.n_pegs - 1):
            self.current_state_parameters.append([])
        key = tuple(tuple(x) for x in self.current_state_parameters)
        self.current_state = self.encoding_to_state[key]
        return self.current_state
        
    def step(self, action) -> Tuple[int, int, int]:
        """Method that performs an action and thereby changes the state of the world.
        The method updates and returns the new state that is the result of the current
        action applied to the current state

        Args:
            action ([type]): Action to be performed on the current state

        Returns:
            Tuple[int, int, int]: A tuple containing the number of the new state, the reward for the
                                    transition, and whether the state is a terminal state 
        """
        enc_next_state = self.perform_action(self.current_state_parameters, action)
        done = self.is_winning(enc_next_state)
        # 10 reward if we're winning, else -1 because we want as few steps as possible
        reward = 10 if done else -1
        self.current_state_parameters = enc_next_state
        tupled_state = tuple(tuple(x) for x in self.current_state_parameters)
        state_number = self.encoding_to_state[tupled_state]
        return state_number, reward, done
    
    def encode_state(self, state: int) -> np.ndarray:
        """Method for encoding a state number in such a way that a neural
        network can process it in a generalized manner.
        In this case, we sum the values of all discs on each peg, such that each
        index results to the sum of the discs on that peg.
        This means that each index corresponds to a peg, and each peg has a value
        between 0 and the sum of all discs 1 -> n

        Args:
            state (int): State number

        Returns:
            np.ndarray: Resulting numpy array from above logic
        """
        encoded_state_tuple = self.state_to_encoding[state]
        encoded_state = np.array([np.sum(x) for x in encoded_state_tuple])
        assert encoded_state.shape == self.enc_shape, f"Shape of state encoding doesn't match encoded state attribute of class ({encoded_state.shape} != {self.enc_shape})"
        return encoded_state
    
    def decode_state(self, encoded_state: np.ndarray) -> int:
        """Method that takes in an encoded state, and turns it into a number

        Args:
            encoded_state (np.ndarray): Aforementioned state

        Returns:
            int: Number corresponding to that state
        """
        encoded_state_tuple = tuple(tuple(x) for x in encoded_state)
        state = self.encoding_to_state[encoded_state_tuple]
        return state

    def get_legal_actions(self, state: int) -> List:
        """Method used during the training-loop that takes in a state number and returns the legal actions.
        Very similar to below method; they are exactly identical, only that this takes in a number, and
        the below an array. Could easily be improved and refactored, but I'm too lazy

        Args:
            state (int): State number

        Returns:
            List: List of tuples, where a tuple is an action
        """
        tuple_enc_state = self.state_to_encoding[state]
        state = list(list(x) for x in tuple_enc_state)
        # Initialize empty list to append valid moves to.
        valid = []
        # Loop through each peg.
        for i in range(self.n_pegs):
            # If peg is empty, ignore it and proceed to next peg.
            if state[i] == []:
                continue
            # Loop through each peg again.
            for j in range(self.n_pegs):
                # If we're not comparing the same pegs.
                if state[i] != state[j]:
                    # Check if j is empty. If yes, then insert [i,j].
                    if len(state[j]) == 0:
                        valid.append((i+1, j+1))
                    # Else, check if disc in peg j is greater than disc in peg i. If yes, then insert.
                    else:
                        if state[j][0] > state[i][0]:
                            valid.append((i+1, j+1))
        return valid

    def generate_legal_actions(self, enc_state: List) -> List:
        """See above docstring
        """
        # Initialize empty list to append valid moves to.
        valid = []
        # Loop through each peg.
        for i in range(self.n_pegs):
            # If peg is empty, ignore it and proceed to next peg.
            if enc_state[i] == []:
                continue
            # Loop through each peg again.
            for j in range(self.n_pegs):
                # If we're not comparing the same pegs.
                if enc_state[i] != enc_state[j]:
                    # Check if j is empty. If yes, then insert [i,j].
                    if len(enc_state[j]) == 0:
                        valid.append((i+1, j+1))
                    # Else, check if disc in peg j is greater than disc in peg i. If yes, then insert.
                    else:
                        if enc_state[j][0] > enc_state[i][0]:
                            valid.append((i+1, j+1))
        return valid

    def perform_action(self, enc_state: List, action: tuple) -> List:
        """Method that takes in an encoded state, and performs an action in the
        form of a tuple

        Args:
            enc_state (List): Encoding of the state in question
            action (tuple): Action to be performed on the state

        Returns:
            List: Encoding of the resulting state
        """
        enc_next_state = copy.deepcopy(enc_state)
        # Get the pegs from and to which the moves are to be done.
        source = action[0]
        dest = action[1]
        # Get the disk from source
        disk = enc_next_state[source - 1][0]
        # Remove from source
        enc_next_state[source-1].remove(disk)
        # Move disk to destination
        enc_next_state[dest - 1].insert(0, disk)
        return enc_next_state
    
    def is_winning(self, enc_state: List) -> bool:
        """Check whether or not the state is winning, which corresponds
        to all discs being small-to-large ordered on the final peg

        Args:
            enc_state (List): Encoding of the state in question

        Returns:
            bool: Whether state is winning or not
        """
        return enc_state[-1] == list(range(1, self.n_discs+1))

    def generate_all_states(self) -> None:
        """Method for generating all possible states in the Hanoi-problem.
        Takes a fair bit of time with 5 pegs and 6 discs due to the sheer number of
        ways the states can be manipulated in that scenario.
        """
        enc_state = self.current_state_parameters
        explored_states = []
        states_to_explore = [enc_state]
        while True:
            for state in states_to_explore:
                legal_moves = self.generate_legal_actions(state)
                for action in legal_moves:
                    new_state = self.perform_action(state, action)
                    if new_state not in self.encoded_states:
                        self.encoded_states.append(new_state)
                        states_to_explore.append(new_state)
                explored_states.append(state)
                states_to_explore.remove(state)
            if len(states_to_explore) == 0:
                break
        
    def print_state(self, state: int) -> None:
        """Method for printing the encoded state in a visually appealing way

        Args:
            enc_state (List): Encoding of the state
        """
        enc_state = list(list(x) for x in self.state_to_encoding[state])
        statenew = copy.deepcopy(enc_state)
        # Iterate through statenew and insert blanks in any position without a peg.
        for i in statenew:
            max_length = self.n_discs
            for k in range(max_length):
                if len(i) == k:
                    for _ in range(max_length - k):
                        i.insert(0, ' ')
        # Loop through a transpose of 'state' and print out the values neatly.
        for i in list(zip(*statenew)):
            for j in i:
                print(str(j) + "|", end=' ')
            print()
        print('----------------------\n')


if __name__ == "__main__":
    game = Hanoi(n_pegs=4, n_discs=4)
    print(game.current_state)
    print(game.current_state_parameters)
    print(game.decode_state(game.current_state_parameters))
    print(game.encode_state(game.current_state))
    print(game.enc_shape)