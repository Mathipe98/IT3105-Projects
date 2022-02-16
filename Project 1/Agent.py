from ctypes import Union
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from cart_pole import CartPoleGame
from hanoi import Hanoi
from gambler import Gambler
from actor import Actor
from critic import Critic

import sys

np.random.seed(123)


class Agent:
    """Hello
    """

    def __init__(self, n_episodes: int, max_steps: int,
                 use_nn: bool, network_dim: Tuple, train_interval: int,
                 a_alpha: float, c_alpha: float, gamma: float, lamb: float,
                 epsilon_start: float, epsilon_finish: float, display: Any, frame_delay: float,
                 chosen_game: int,
                 game: Any) -> None:
        """Constructor for the overlooking RL Agent in the Actor-Critic paradigm. The Agent is the "middleman" between
        the Actor and the Critic, and is the one telling them what to do at any given time.

        Args:
            n_episodes (int): Number of episodes for the agent to run the training for
            max_steps (int): Max number of steps allowed before timing out
            use_nn (bool): Whether to use a neural network critic
            network_dim (Tuple): Dimensions of neural network if used
            train_interval (int): The interval of episodes at which the network will be trained (every n steps)
            a_alpha (float): Learning rate of the actor
            c_alpha (float): Learning rate of the critic
            gamma (float): Future reward discount rate (same for actor and critic)
            lamb (float): Eligibility trace decay rate (same for actor and critic)
            epsilon_start (float): The epsilon value to start with in the decayed-epsilon-greedy strategy
            epsilon_finish (float): The epsilon
            display (Any): Whether to display information about the learning process and the results
            frame_delay (float): Delay between frames when displaying the run of a game (only relevant for Hanoi)
            chosen_game (int): Number that corresponds to the chosen game (1-3). NB: ONLY RELEVANT FOR VISUALIZATION; is in no way used in the search algorithm
            game (Any): Object corresponding to the chosen game. Is used in a generic and general way.
        """
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.use_nn, self.network_dim = use_nn, network_dim
        self.train_interval = train_interval
        self.a_alpha, self.c_alpha = a_alpha, c_alpha
        self.gamma, self.lamb, = gamma, lamb
        self.epsilon_start, self.epsilon_finish = epsilon_start, epsilon_finish
        self.display, self.frame_delay = display, frame_delay
        self.chosen_game = chosen_game
        self.game = game
        self.actor = Actor(a_alpha, gamma, lamb, epsilon_start, game=self.game)
        self.critic = Critic(c_alpha, gamma, lamb, game=self.game,
                             use_nn=use_nn, network_dim=network_dim)
        # Create sets to keep track of number of steps, actions, and what starting state for every episode (for logging)
        self.steps = {}
        self.actions = {}
        self.starting_states = {}

    def reset_eligibilities(self) -> None:
        """Method for resetting the eligibility values of the actor and critic when used for eligibility traces.
        Note that the critic does NOT use eligibility traces when using a neural network.
        """
        self.actor.reset_eligibilities()
        if not self.use_nn:
            self.critic.reset_eligibilities()

    def train(self) -> None:
        """Main method for the actor-critic training loop. What's happening, is the following:
            - First we create a queue (list with max length) that contains examples of previously
                observed data. This is only used for the neural network (if used)
            - We iterate for n episodes, and for each episode we perform some step in the simworld
            - Every step of the simworld leads to a new state with new internal variables (not seen to the actor/critic;
                only state itself is seen)
            - With this newly observed information, we use the bellman-equations for optimality to tune the value and policy
                functions respectively where the assumed "correct" value is the discounted future value of each state/SAP
            - We then use this assumed correct value as a correcting term, and tune the value-/policy-function by propagating it
                backwards with eligibility traces
            - This is due to the assumption that a state alone isn't necessarily the alone culprit of winning or losing; it's most
                likely a sequence of actions that yielded the current state, and thus any reward (or penalty) should be propagated
                backwards so that other states also somewhat gain the effect of the current state
            - We then perform this sequence of operations for all episodes, and the algorithm will then (with some combination of hyperparameters)
                hopefully converge to a correct result
        """
        replay_memory = deque(maxlen=50000)

        if self.display:
            progress = 0
            progress_bar = "[          ]"
            sys.stdout.write("\tProgress: %d%% " % (progress) + progress_bar + "   \r")
            sys.stdout.flush()
            next_progress = progress + int(self.n_episodes * 0.1)
        for episode in range(self.n_episodes):
            # At the start of every episode, we decay epsilon ever so slightly
            r = max((self.n_episodes - episode) / self.n_episodes, 0)
            decayed_epsilon = (self.epsilon_start -
                               self.epsilon_finish) * r + self.epsilon_finish
            self.actor.epsilon = decayed_epsilon
            self.reset_eligibilities()
            current_state = self.game.reset()
            current_action = self.actor.get_action(current_state)
            current_steps = 0
            current_episode_actions = []
            current_episode_start_params = self.game.current_state_parameters
            self.starting_states[episode] = current_episode_start_params
            visited = []
            done = False
            while not done:
                current_episode_actions.append(current_action)
                next_state, reward, done = self.game.step(current_action)
                # print(next_state)

                next_action = self.actor.get_action(next_state)
                current_SAP = (current_state, current_action)
                if current_SAP not in visited:
                    visited.append(current_SAP)
                next_SAP = (next_state, next_action)
                self.actor.set_eligibility(current_SAP)
                if self.use_nn:
                    shape = self.game.enc_shape
                    encoded_current_state = self.game.encode_state(
                        current_state)
                    encoded_next_state = self.game.encode_state(next_state)
                    # Reshape in prediction because the model expects input dimensions (batch_size, length of encoded state)
                    current_state_prediction = self.critic.evaluate_state(
                        encoded_current_state.reshape(1, shape[0]))
                    next_state_prediction = self.critic.evaluate_state(
                        encoded_next_state.reshape(1, shape[0]))
                else:
                    self.critic.set_eligibility(current_SAP)
                    current_state_prediction = self.critic.values[current_SAP]
                    next_state_prediction = self.critic.values[next_SAP]
                # Calculate delta and propagate it backwards, but if we're done then stop propagation
                if done:
                    delta = reward
                else:
                    target = reward + self.gamma * next_state_prediction
                    delta = target - current_state_prediction
                for SAP in visited:
                    self.actor.update(SAP, delta)
                    self.actor.update_eligibility(SAP)
                    if not self.use_nn:
                        self.critic.table_update(SAP, delta)
                        self.critic.update_eligibility(SAP)
                if self.use_nn:
                    replay_memory.append(
                        [encoded_current_state, target, reward, done])
                    # Perform training every n steps, and only if we've encountered sufficient number of previous examples
                    if current_steps % self.train_interval == 0 and len(replay_memory) > 1000:
                        self.critic.train(replay_memory)
                current_state = next_state
                current_action = next_action
                current_steps += 1
                if done or current_steps >= self.max_steps:
                    self.steps[episode] = current_steps
                    self.actions[episode] = current_episode_actions
                    break
            if self.display:
                progress = (episode + 1)
                if progress == next_progress:
                    percent = int((progress / self.n_episodes) * 100)
                    index = int(percent / 10)
                    pbar_list = list(progress_bar)
                    pbar_list[index] = "="
                    progress_bar = ''.join(pbar_list)
                    sys.stdout.write("\tProgress: %d%% " % (percent) + progress_bar + "   \r")
                    sys.stdout.flush()
                    next_progress = progress + int(self.n_episodes * 0.1)

    def visualize(self) -> None:
        """Method that visualizes the relevant information of each game.
        This is the only place where we use the chosen_game variable, seeing
        as the different games have different requirements for visualization.
        Therefore I used this variable only as a simple way to distinguish what
        to visualize.

        Also note that the Towers of Hanoi is the only game that actually runs
        a game; the other ones only produce plots (and print statements).

        Games:
            1 = Cartpole
            2 = Hanoi
            3 = Gambler
        """
        print("\n\n")
        if self.chosen_game == 1:
            self.visualize_steps()
            self.visualize_cartpole()
        elif self.chosen_game == 2:
            self.visualize_steps()
            self.run_game()
        else:
            self.visualize_gambler()

    def visualize_steps(self) -> None:
        """Simple method for generating a plot with number of steps as a function of number of episodes
        """
        plt.plot(list(self.steps.values()))
        plt.xlabel('N episodes')
        plt.ylabel("Steps")
        plt.show()
    
    def visualize_cartpole(self) -> None:
        """Specific method for visualizing the states of the Cartpole-game
        """
        best_episode = max(self.actions, key=lambda key: len(self.actions[key]))
        best_actions = self.actions[best_episode]
        self.game.reset()
        self.game.current_state_parameters = self.starting_states[best_episode]
        angles = [self.game.current_state_parameters[2]]
        for action in best_actions:
            self.game.step(action)
            angles.append(self.game.current_state_parameters[2])
        plt.plot(angles)
        plt.xlabel("N steps")
        plt.ylabel("Angle (radians)")
        plt.show()
    
    def visualize_gambler(self) -> None:
        self.actor.epsilon = 0
        states = self.game.states
        max_values = {}
        for state in states:
            if state == 0 or state == 100:
                continue
            max_values[state] = self.actor.get_action(state)
        plt.plot(list(max_values.values()))
        plt.xlabel("States")
        plt.ylabel("Action")
        plt.show()
    
    def run_game(self) -> None:
        """Method for running through a game and producing/executing an action
        for each encountered state.
        Method is generic and works for all games, but really only makes sense to
        use for the Hanoi game because it can actually print the states in a nice
        way. In the other cases, it just prints the state parameters directly, which
        isn't that nice to look at.
        """
        current_state = self.game.reset()
        self.actor.epsilon = 0
        current_action = self.actor.get_action(current_state)
        steps = 0
        done = False
        while not done:
            steps += 1
            next_state, _, done = self.game.step(current_action)
            next_action = self.actor.get_action(next_state)
            if self.chosen_game == 2:
                self.game.print_state(current_state)
            else:
                print(self.game.current_state_parameters)
            time.sleep(self.frame_delay)
            current_state = next_state
            current_action = next_action
        if self.chosen_game == 2:
            self.game.print_state(next_state)
        else:
            print(self.game.current_state_parameters)
        print(f"Length of run: {steps}")
