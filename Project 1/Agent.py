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
                 a_alpha: float, c_alpha: float, gamma: float,
                 epsilon: float, display: Any, frame_delay: float,
                 game: CartPoleGame) -> None:
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.use_nn, self.network_dim = use_nn, network_dim
        self.a_alpha, self.c_alpha = a_alpha, c_alpha
        self.gamma, self.epsilon = gamma, epsilon
        self.display, self.frame_delay = display, frame_delay
        self.game = game
        self.n_states = self.game.n_states
        self.n_actions = self.game.n_actions
        self.actor = Actor(epsilon, a_alpha, gamma, self.n_states, self.n_actions)
        self.critic = Critic(c_alpha, gamma, self.n_states, self.n_actions, use_nn)

    def reset_eligibilities(self) -> None:
        """blabla
        """
        pass

    def start_training(self) -> None:
        """Bla
        """
        self.critic.initialize()
        self.actor.initialize()
        self.game.start_episode()
        for i in range(self.n_episodes):
            self.reset_eligibilities()
            s_init = self.game.current_state
            a_init = self.actor.get_best_action(s_init)
            current_state = self.game.current_state
            next_state, reward = self.game.generate_child_state(a_init)
            next_action = self.actor.get_best_action(next_state)
            delta = self.critic.calculate_delta(
                r=reward, s1=current_state, s2=next_state, a1=a_init, a2=next_action)
            
            


def test():
    """blabla
    """
    global_config = {
        "n_episodes": 20,
        "max_steps": 1,
        "use_nn": False,
        "network_dim": None,
        "a_alpha": 0.1,
        "c_alpha": 0.1,
        "gamma": 0.9,
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
