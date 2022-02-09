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

    def reset_eligibilities(self) -> None:
        """blabla
        """
        self.critic.reset_eligibilities()
        self.actor.reset_eligibilities()

    def start_training(self) -> None:
        """Bla
        """
        for episode in range(self.n_episodes):
            self.reset_eligibilities()
            self.game.reset()
            steps = 0
            current_state = self.game.current_state
            current_action = self.actor.get_action(current_state)
            # Keep track of the state-action-pairs we've visited in this episode
            visited_SAPs = []
            # Now start the step-iteration
            while steps < self.max_steps:
                next_state, reward, done = self.game.step(current_action)
                next_action = self.actor.get_action(next_state)
                visited_SAPs.append((current_state, current_action))
                delta = self.critic.calculate_delta(
                    r=reward, s1=current_state, s2=next_state, a1=current_action, a2=next_action)
                # Set the eligibility of the current SAP to 1
                self.critic.set_eligibility(current_state, current_action)
                self.actor.set_eligibility(current_state, current_action)
                for state, action in visited_SAPs:
                    self.critic.update(delta, state, action)
                    self.actor.update(delta, state, action)
                    # If s_t != s, then update. We've already dealt with s_t = t (above; set_eligibility)
                    if state != current_state:
                        self.critic.update_eligibility(state, action)
                        self.actor.update_eligibility(state, action)
                # Update the game s.t. the next state produced is the current state
                self.game.update_state()
                current_state, current_action = next_state, next_action
                steps += 1
                if done:
                    break
            print(f"Episode number:\t {episode}\nSteps:\t {steps}")
            


def test():
    """blabla
    """
    global_config = {
        "n_episodes": 2000,
        "max_steps": 1000,
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
        "m_p": 1,
        "l": 0.5,
        "tau": 0.02,
        "n_tilings": 1,
        "feat_bins": None
    }
    world = CartPoleGame(**cart_pole_config)
    system = RLSystem(**global_config, game=world)
    system.start_training()



if __name__ == '__main__':
    test()
