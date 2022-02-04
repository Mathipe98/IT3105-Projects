from ctypes import Union
from typing import Any, Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from cart_pole import CartPoleGame
from actor import Actor
from critic import Critic

class RLSystem:

    def __init__(self, n_episodes: int, td_steps: int,
                 use_nn: bool, network_dim: Tuple,
                 act_crit_params: List[float],
                 # act_lr: float, act_decay: float, act_disc: float,
                 # crit_lr: float, crit_decay: float, crit_disc: float,
                 epsilon: float, display: Any, frame_delay: float,
                 sim_world: CartPoleGame) -> None:
        self.n_episodes = n_episodes
        self.td_steps = td_steps
        self.use_nn = use_nn
        self.network_dim = network_dim
        self.act_crit_params = act_crit_params
        self.epsilon = epsilon
        self.display = display
        self.frame_delay = frame_delay
        self.sim_world = sim_world
        # self.actor = Actor(**actor_params)
        # self.critic = Critic(**critic_params)
        self.actor = Actor(sim_world, epsilon)
        self.critic = Critic(sim_world, actor=self.actor)

    def reset_eligibilities(self) -> None:
        pass

    def actor_critic_algorithm(self) -> None:
        self.critic.initialize()
        self.actor.initialize()
        for i in range(self.n_episodes):
            self.reset_eligibilities()
            s_init = self.sim_world.current_state
            s_init_index = self.actor.get_state_index(s_init)
            print(s_init)
            print(f"Starting theta: {self.sim_world.state_parameters[2]}")
            while True:
                a_prime = self.actor.get_best_action(s_init_index)


def test():
    global_config = {
        "n_episodes": 1,
        "td_steps": 1,
        "use_nn": False,
        "network_dim": None,
        "act_crit_params": [0.1],
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
    system = RLSystem(**global_config, sim_world=world)
    system.actor_critic_algorithm()





if __name__ == '__main__':
    test()
    