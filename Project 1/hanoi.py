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
        self.n_states = None
        self.state_to_encoding = {}
        self.encoding_to_state = {}
        self.counter = 0
        self.initialize()
    
    def initialize(self) -> None:
        self.current_state_parameters = []
        self.current_state_parameters.append(list(range(1,self.n_discs+1)))
        for _ in range(self.n_pegs - 1):
            self.current_state_parameters.append([])
        self.generate_all_states()
        self.n_states = len(self.encoded_states)
        assert len(self.encoded_states) != 0, "States not generated"
        for state_number in range(len(self.encoded_states)):
            encoded_state_tuple = tuple(tuple(x) for x in self.encoded_states[state_number])
            self.state_to_encoding[state_number] = encoded_state_tuple
            self.encoding_to_state[encoded_state_tuple] = state_number
            self.states.append(state_number)
        key = tuple(tuple(x) for x in self.current_state_parameters)
        self.current_state = self.encoding_to_state[key]

    def reset(self) -> None:
        self.current_state_parameters = []
        self.current_state_parameters.append(list(range(1,self.n_discs+1)))
        for _ in range(self.n_pegs - 1):
            self.current_state_parameters.append([])
        key = tuple(tuple(x) for x in self.current_state_parameters)
        self.current_state = self.encoding_to_state[key]
        
    def step(self, action) -> Tuple[int, int, int]:
        enc_next_state = self.perform_action(self.current_state_parameters, action)
        done = self.is_winning(enc_next_state)
        # 10 reward if we're winning, else -1 because we want as few steps as possible
        reward = 10 if done else -1
        self.current_state_parameters = enc_next_state
        tupled_state = tuple(tuple(x) for x in self.current_state_parameters)
        state_number = self.encoding_to_state[tupled_state]
        return state_number, reward, done

    def get_legal_actions(self, state: int) -> List:
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
        return enc_state[-1] == list(range(1, self.n_discs+1))

    def generate_all_states(self) -> None:
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
        
    def print_state(self, enc_state: List) -> None:
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
        print('----------------------')


if __name__ == "__main__":
    game = Hanoi(n_pegs=3, n_discs=3)
    print("hello")
    print(game.counter)
    
