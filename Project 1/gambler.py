from typing import List, Tuple
import numpy as np


class Gambler:

    def __init__(self, p_win: float=0.5, goal_cash: int=100) -> None:
        """Constructor for the Gambler's ruin problem

        Args:
            p_win (float, optional): Probability of winning the coin toss. Defaults to 0.5.
            goal_cash (int, optional): The amount of money needed to win. Defaults to 100.
        """
        self.p_win = p_win
        self.goal_cash = goal_cash
        self.states = None
        self.current_state = None
        self.current_state_parameters = None
        self.enc_shape = None
        self.initialize()
    
    def initialize(self) -> None:
        """Method for initializing the game by setting necessary parameters and
        generating all the possible states.
        """
        self.generate_all_states()
        self.reset()
        self.enc_shape = (1,)

    def reset(self) -> None:
        """Method for resetting the game parameters when starting a new episode

        Returns:
            _type_: State number of the reset state
        """
        self.current_state = np.random.randint(1, self.goal_cash-1)
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, int, int]:
        """Method that takes an action, and uses this action along with its own
        internal parameters to produce a new state along with some reward depending
        on the outcome of the action.

        Args:
            action (int): Integer describing the action

        Returns:
            Tuple[int, int, int]: Tuple containing the next state, the reward from the transition,
                                    and whether the next state is terminal
        """
        if np.random.uniform(0,1) < self.p_win:
            next_state = min(self.goal_cash, self.current_state + action)
        else:
            next_state = max(0, self.current_state - action)
        if next_state == self.goal_cash:
            reward = 10
            done = True
        else:
            reward = 0
            done = next_state == 0
        self.current_state = next_state
        return next_state, reward, done

    def encode_state(self, state: int) -> np.ndarray:
        """Method for encoding a state number into a numpy array.
        It's used for the neural network, however there really isn't
        a good way to encode it since the problem is inherently random,
        so we just return an array of the number (because the network needs
        either tensors or arrays as input)

        Args:
            state (int): Number of the state

        Returns:
            np.ndarray: Array of the same number
        """
        return np.array([state])

    def decode_state(self, encoded_state: np.ndarray) -> int:
        """Opposite of above method; extracts the number from an array.

        Args:
            encoded_state (np.ndarray): Array containing the state

        Returns:
            int: Number of the (encoded) state
        """
        return encoded_state[0]

    def get_legal_actions(self, state: int) -> np.ndarray:
        """Method that returns an array of legal actions for the current state.
        Something worth noting is that for the indexing and the generic actor-critic
        algorithm to work, we must provide some legal actions for states 0 and 100 (even
        though these are terminal states). However their values will never be used (because the 
        algorithm doesn't propagate rewards of terminal states). So it's simply a way to avoid crashes.

        Args:
            state (int): Number of the state in question

        Returns:
            np.ndarray: Array containing all legal actions for that state
        """
        if state == 0 or state == 100:
            return np.array([0])
        end = min(state, self.goal_cash - state) + 1
        return np.array([cash_spent for cash_spent in range(1, end)])

    def is_winning(self, state: int) -> bool:
        """Method that checks whether a state is winning

        Args:
            state (int): State number

        Returns:
            bool: Result of check
        """
        return state == self.goal_cash

    def generate_all_states(self) -> None:
        """Method that generates all states for this game
        """
        self.states = np.array([i for i in range(0, self.goal_cash + 1)])
