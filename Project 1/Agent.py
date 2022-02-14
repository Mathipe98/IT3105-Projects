from ctypes import Union
from typing import Any, Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from cart_pole import CartPoleGame
from actor import Actor
from critic import Critic
from hanoi import Hanoi


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
        # self.n_actions = self.game.n_actions
        self.actor = Actor(a_alpha, gamma, lamb, epsilon, game=self.game)
        self.critic = Critic(c_alpha, gamma, lamb, game=self.game, use_nn=use_nn, network_dim=network_dim)
        self.steps = {}
        self.actions = {}
        self.angles = []

    def reset_eligibilities(self) -> None:
        """blabla
        """
        self.critic.reset_eligibilities()
        self.actor.reset_eligibilities()

    def train(self) -> None:
        """Bla
        """
        for episode in range(self.n_episodes):
            self.reset_eligibilities()
            self.game.reset()
            current_steps = 0
            current_episode_actions = []
            current_state = self.game.current_state
            current_action = self.actor.get_action(current_state)
            # Now start the step-iteration
            visited = []
            done = False
            print(f"Episode number: {episode}")
            while not done:
                current_episode_actions.append(current_action)
                next_state, reward, done = self.game.step(current_action)
                next_action = self.actor.get_action(next_state)
                if not self.use_nn:
                    current_SAP = (current_state, current_action)
                    if current_SAP not in visited:
                        visited.append(current_SAP)
                    next_SAP = (next_state, next_action)
                    self.critic.set_eligibility(current_SAP)
                    self.actor.set_eligibility(current_SAP)
                    # Calculate delta and propagate it backwards
                    delta = reward + self.gamma * self.actor.PI[next_SAP] - self.actor.PI[current_SAP]
                    for SAP in visited:
                        self.actor.update(SAP, delta)
                        self.critic.table_update(SAP, delta)
                        self.actor.update_eligibility(SAP)
                        self.critic.update_eligibility(SAP)
                else:
                    pass
                current_state = next_state
                current_action = next_action
                current_steps += 1
                if done or current_steps >= self.max_steps:
                    self.steps[episode] = current_steps
                    self.actions[episode] = current_episode_actions
                    break

    def Q_learning(self) -> None:
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
                Q1 = self.actor.PI[s][a]
                Q2 = np.max(self.actor.PI[s_next])
                self.actor.PI[s][a] += self.a_alpha * (reward + self.gamma * Q2 - Q1)

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
        plt.plot(list(self.steps.values()))
        plt.xlabel('N episodes')
        plt.ylabel("Steps")
        plt.show()
    
    def visualize_hanoi_sequence(self, actions: List[tuple]) -> None:
        self.game.reset()
        start = self.game.current_state_parameters
        print("\n")
        self.game.print_state(start)
        for action in actions:
            print("\n")
            self.game.step(action)
            next_state = self.game.current_state_parameters
            self.game.print_state(next_state)
    
    def visualize_angles(self) -> None:
        plt.plot(self.angles)
        plt.xlabel("N steps")
        plt.ylabel("Angle (radians)")
        plt.show()


def test_cartpole():
    """blabla
    """
    global_config = {
        "n_episodes": 300,
        "max_steps": 300,
        "use_nn": True,
        "network_dim": (15,30,20,5,1),
        "a_alpha": 0.08,
        "c_alpha": 0.08,
        "gamma": 0.9,
        "lamb": 0.99,
        "epsilon": 0.0,
        "display": 0,
        "frame_delay": 0
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
    system = RLSystem(**global_config, game=world)
    system.train()
    system.visualize_steps()
    # system.visualize_angles()

def test_hanoi() -> None:
    global_config = {
        "n_episodes": 1000,
        "max_steps": 100,
        "use_nn": False,
        "network_dim": (100, 50, 25, 30, 1),
        "a_alpha": 0.1,
        "c_alpha": 0.1,
        "gamma": 0.9,
        "lamb": 0.99,
        "epsilon": 0.0,
        "display": 0,
        "frame_delay": 0
    }
    game = Hanoi(n_pegs=5, n_discs=6)
    system = RLSystem(**global_config, game=game)
    system.train()
    system.visualize_steps()
    best_episode = min(system.actions, key=lambda key: len(system.actions[key]))
    best_actions = system.actions[best_episode]
    print(best_actions)
    system.visualize_hanoi_sequence(best_actions)


if __name__ == '__main__':
    test_hanoi()
    # test_cartpole()