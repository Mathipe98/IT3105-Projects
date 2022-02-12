from ctypes import Union
from typing import Any, Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from cart_pole import CartPoleGame
from actor import Actor
from critic import Critic

# act_lr: float, act_decay: float, act_disc: float,
# crit_lr: float, crit_decay: float, crit_disc: float,


class RLSystem:
    """Hello
    """

    def __init__(self, n_episodes: int, max_steps: int,
                 use_nn: bool, network_dim: Tuple,
                 a_alpha: float, c_alpha: float, gamma: float, lamb: float,
                 epsilon: float, display: Any, frame_delay: float,
                 game: CartPoleGame) -> None:
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.use_nn, self.network_dim = use_nn, network_dim
        self.a_alpha, self.c_alpha = a_alpha, c_alpha
        self.gamma, self.lamb, self.epsilon = gamma, lamb, epsilon
        self.display, self.frame_delay = display, frame_delay
        self.game = game
        self.n_states = self.game.n_states
        self.n_actions = self.game.n_actions
        self.actor = Actor(a_alpha, gamma, lamb, epsilon, self.n_states, self.n_actions)
        self.critic = Critic(c_alpha, gamma, lamb, self.n_states, self.n_actions, use_nn)
        self.steps = []
        self.angles = []

    def reset_eligibilities(self) -> None:
        """blabla
        """
        self.critic.reset_eligibilities()
        self.actor.reset_eligibilities()

    def start_training(self) -> None:
        """Bla
        """
        best_steps = 0
        best_angles = []
        for episode in range(self.n_episodes):
            self.reset_eligibilities()
            self.game.reset()
            steps = 0
            current_state = self.game.current_state
            current_action = self.actor.get_action(current_state)
            # Now start the step-iteration
            print(episode)
            temp = []
            temp.append(self.game.current_state_parameters[2])
            visited = []
            while steps < self.max_steps:
                next_state, reward, done = self.game.step(current_action)
                temp.append(self.game.current_state_parameters[2])
                next_action = self.actor.get_action(next_state)
                delta = self.critic.calculate_delta(
                    r=reward, s1=current_state, s2=next_state, a1=current_action, a2=next_action)
                # Set the eligibility of the current SAP to 1
                self.critic.set_eligibility(current_state, current_action)
                self.actor.set_eligibility(current_state, current_action)
                self.actor.update(delta, current_state, current_action)
                self.critic.update(delta, current_state, current_action)
                # Iterate through all state-action-pairs
                for state,action in visited:
                    self.critic.update(delta, state, action)
                    self.actor.update(delta, state, action)
                    self.critic.update_eligibility(state, action)
                    self.actor.update_eligibility(state, action)
                current_state, current_action = next_state, next_action
                #if (current_state, current_action) not in visited:
                visited.append((current_state, current_action))
                steps += 1
                if done:
                    if steps > best_steps:
                        best_angles = temp
                        best_steps = steps
                    self.steps.append(steps)
                    break
            self.angles = best_angles

    def SARSA(self) -> None:
        best_steps = 0
        best_angles = []
        for episode in range(self.n_episodes):
            temp = []
            self.game.reset()
            steps = 0
            s = self.game.current_state
            a = self.actor.get_action(s)
            temp.append(self.game.current_state_parameters[2])
            while steps < self.max_steps:
                s_next, reward, done = self.game.step(a)
                a_next = self.actor.get_action(s_next)
                Q1 = self.actor.PI[s, a]
                Q2 = self.actor.PI[s_next, a_next]
                self.actor.PI[s, a] = Q1 + self.a_alpha * (reward + self.gamma * Q2 - Q1)

                temp.append(self.game.current_state_parameters[2])
                s, a = s_next, a_next
                steps += 1
                if done:
                    if steps > best_steps:
                        best_angles = temp
                        best_steps = steps
                    self.steps.append(steps)
                    break
            self.angles = best_angles
            

    def visualize_steps(self) -> None:
        plt.plot(self.steps)
        plt.xlabel('N episodes')
        plt.ylabel("Steps")
        plt.show()
    
    def visualize_angles(self) -> None:
        plt.plot(self.angles)
        plt.xlabel("N steps")
        plt.ylabel("Angle (radians)")
        plt.show()


def test():
    """blabla
    """
    global_config = {
        "n_episodes": 500,
        "max_steps": 300,
        "use_nn": False,
        "network_dim": None,
        "a_alpha": 0.1,
        "c_alpha": 0.1,
        "gamma": 0.9,
        "lamb": 0.99,
        "epsilon": 0.1,
        "display": 0,
        "frame_delay": 0
    }
    cart_pole_config = {
        "g": 9.81,
        "m_c": 1,
        "m_p": 0.5,
        "l": 0.5,
        "tau": 0.01,
        "n_tilings": 1,
        "feat_bins": None
    }
    world = CartPoleGame(**cart_pole_config)
    system = RLSystem(**global_config, game=world)
    system.SARSA()
    system.visualize_steps()
    # system.visualize_angles()


if __name__ == '__main__':
    test()
