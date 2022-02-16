from typing import Any
import numpy as np
import random

np.random.seed(123)

class Actor:

    def __init__(self, alpha: float, gamma: float, lamb: float,
                epsilon: float, game: Any) -> None:
        """Constructor for the Actor in the Actor-Critic algorithm.

        Args:
            alpha (float): Learning rate of the actor
            gamma (float): Future reward discount rate
            lamb (float): Eligibility trace decay rate
            epsilon (float): The probability of performing a random action in epsilon-greedy strategy
            game (Any): The game to play. Communcatin happens via generic functions and methods
        """
        self.alpha, self.gamma, self.lamb = alpha, gamma, lamb
        self.epsilon = epsilon
        self.PI = {}
        self.eligibility_trace = {}
        self.debug = 0
        self.game = game
        self.initialize()
    
    def initialize(self) -> None:
        """Method for initializing the Actor by setting up the 
        policy map (which uses SAPs) and the same for eligibility traces.
        """
        states = self.game.states
        for state in states:
            actions = self.game.get_legal_actions(state)
            for action in actions:
                SAP = (state, action)
                self.PI[SAP] = 0
                self.eligibility_trace[SAP] = 0
        pass

    def update(self, SAP: tuple, delta: float) -> None:
        """Method for updating the policy map of a state-action pair.

        Args:
            SAP (tuple): The state-action pair
            delta (float): The delta (error) term used to adjust the value
        """
        self.PI[SAP] = self.PI[SAP] + self.alpha * delta * self.eligibility_trace[SAP]

    def set_eligibility(self, SAP: tuple) -> None:
        """Method for setting the eligibility value of 1 to the
        current state-action pair. Here we do not use accumulative
        eligibility traces, but rather the max variant

        Args:
            SAP (tuple): State-action pair
        """
        self.eligibility_trace[SAP] = 1

    def update_eligibility(self, SAP: tuple) -> None:
        """Updating the eligibility trace of the current 
        state-action pair in a decaying manner

        Args:
            SAP (tuple): State-action pair
        """
        self.eligibility_trace[SAP] = self.gamma * \
            self.lamb * self.eligibility_trace[SAP]

    def reset_eligibilities(self) -> None:
        """Method for resetting all eligibility traces
        when a new episode starts.
        """
        states = self.game.states
        for state in states:
            actions = self.game.get_legal_actions(state)
            for action in actions:
                SAP = (state, action)
                self.eligibility_trace[SAP] = 0
    
    def get_action(self, state: int) -> int:
        """Method that takes in a state number, and returns the
        best action according to the policy, which corresponds to:
        action a = argmax(PI[s,a])

        Args:
            state (int): Number representation of the state

        Returns:
            int: The best action according to the actor's policy
        """
        actions = self.game.get_legal_actions(state)
        if np.random.uniform(0,1) < self.epsilon:
             return random.choice(actions)
        possible_SAPs = {(state, action): self.PI[(state, action)] for action in actions}
        return max(possible_SAPs, key=possible_SAPs.get)[1]
