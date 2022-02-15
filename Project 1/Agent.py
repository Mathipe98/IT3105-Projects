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

np.random.seed(123)


class Agent:
    """Hello
    """

    def __init__(self, n_episodes: int, max_steps: int,
                 use_nn: bool, network_dim: Tuple, train_interval: int,
                 a_alpha: float, c_alpha: float, gamma: float, lamb: float,
                 epsilon_start: float, epsilon_finish: float, display: Any, frame_delay: float,
                 chosen_game: int,
                 game: CartPoleGame) -> None:
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
            print(f"Episode number: {episode}")
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

    def visualize_steps(self) -> None:
        """Simple method for generating a plot with number of steps as a function of number of episodes
        """
        plt.plot(list(self.steps.values()))
        plt.xlabel('N episodes')
        plt.ylabel("Steps")
        plt.show()

    def visualize_hanoi(self) -> None:
        """Specific method for visualizing the states of the Hanoi-game
        """
        best_episode = min(
            self.actions, key=lambda key: len(self.actions[key]))
        best_actions = self.actions[best_episode]
        start = self.game.reset()
        print("\n")
        self.game.print_state(start)
        for action in best_actions:
            print("\n")
            self.game.step(action)
            next_state = self.game.current_state_parameters
            self.game.print_state(next_state)

    def run_game(self) -> None:
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
        states = self.game.states
        critic_values = self.critic.values
        max_values = {}
        for state in states:
            if state == 0 or state == 100:
                continue
            actions = self.game.get_legal_actions(state)
            possible_SAPs = {(state, action): critic_values[(state, action)] for action in actions}
            max_action = max(possible_SAPs, key=possible_SAPs.get)[1]
            max_values[state] = max_action
        plt.plot(list(max_values.values()))
        plt.xlabel("States")
        plt.ylabel("Action")
        plt.show()


def run_hanoi_game(agent: Agent, game: Hanoi, delay: float) -> None:
    current_state = game.reset()
    agent.actor.epsilon = 0
    current_action = agent.actor.get_action(current_state)
    done = False
    while not done:
        next_state, _, done = game.step(current_action)
        next_action = agent.actor.get_action(next_state)
        game.print_state(current_state)
        time.sleep(delay)
        current_state = next_state
        current_action = next_action
    game.print_state(next_state)


def test_cartpole():
    """blabla
    """
    global_config = {
        "n_episodes": 300,
        "max_steps": 300,
        "use_nn": False,
        "network_dim": (50,10,1),
        "a_alpha": 0.1,
        "c_alpha": 0.001,
        "gamma": 0.9,
        "lamb": 0.99,
        "epsilon": 0.1,
        "display": 0,
        "frame_delay": 0,
        "train_interval": 50,
    }
    cart_pole_config = {
        "g": -9.81,
        "m_c": 1,
        "m_p": 0.1,
        "l": 0.5,
        "tau": 0.02,
        "buckets": (4,4,8,8)
    }
    world = CartPoleGame(**cart_pole_config)
    system = Agent(**global_config, game=world)
    system.train()
    system.visualize_steps()
    system.visualize_cartpole()

def test_hanoi() -> None:
    global_config = {
        "n_episodes": 200,
        "max_steps": 100,
        "use_nn": True,
        "network_dim": (24,12,1),
        "a_alpha": 0.1,
        "c_alpha": 0.001,
        "gamma": 0.9,
        "lamb": 0.99,
        "epsilon_start": 1,
        "epsilon_finish": 0.0,
        "display": 0,
        "frame_delay": 0,
        "train_interval": 5,
    }
    hanoi_config = {
        "n_pegs": 3,
        "n_discs": 4,
    }
    game = Hanoi(**hanoi_config)
    system = Agent(**global_config, game=game)
    system.train()
    system.visualize_steps()

    run_hanoi_game(system, game, 1)

def test_gambler() -> None:
    global_config = {
        "n_episodes": 25000,
        "max_steps": 100,
        "use_nn": False,
        "network_dim": (24,12,1),
        "a_alpha": 0.1,
        "c_alpha": 0.1,
        "gamma": 0.9,
        "lamb": 0.99,
        "epsilon": 0.1,
        "display": 0,
        "frame_delay": 0,
        "train_interval": 50,
    }
    gambler_params = {
        "p_win": 0.4,
        "goal_cash": 100,
    }
    gambler = Gambler(**gambler_params)
    agent = Agent(**global_config, game=gambler)
    agent.train()
    agent.visualize_gambler()

if __name__ == '__main__':
    test_hanoi()
    # test_cartpole()
    # test_gambler()
    pass