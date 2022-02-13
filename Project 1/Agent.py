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
                # print(f"Episode number: {episode}")
                # print(f"Number of visited states: {len(visited)}")
                # print(f"Current state: {current_state}")
                # print(f"Current state parameters: {self.game.current_state_parameters}")
                next_state, reward, done = self.game.step(current_action)
                # print(f"Next state: {next_state}")
                # print(f"Next state parameters: {self.game.current_state_parameters}")
                # print(f"Gained reward from taking action: {reward}")
                next_action = self.actor.get_action(next_state)
                # print(f"Next action: {next_action}")
                delta = self.critic.calculate_delta(
                    r=reward, s1=current_state, s2=next_state, a1=current_action, a2=next_action)
                # print(f"Delta value: {delta}")
                # Set the eligibility of the current SAP to 1
                if not self.use_nn:
                    self.critic.set_eligibility(current_state, current_action)
                    self.actor.set_eligibility(current_state, current_action)
                    if (current_state, current_action) not in visited:
                        visited.append((current_state, current_action))
                    # Iterate through all state-action-pairs
                    # print(f"Visited states: {[state[0] for state in visited]}")
                    for state,action in visited:
                        self.critic.update(state, action, delta)
                        self.actor.update(state, action, delta)
                        self.critic.update_eligibility(state, action)
                        self.actor.update_eligibility(state, action)
                else:
                    tup = (current_state, current_action, next_state, next_action, reward)
                    if tup not in visited:
                        visited.append(tup)
                    # Iterate through all state-action-pairs
                    # print(f"Visited states: {[state[0] for state in visited]}")
                    for s1,a1,_,_,_ in visited:
                        self.actor.update(s1, a1, delta)
                # print(f"Current state before assignment: {current_state}")
                # print(f"Next state before assignment: {next_state}")
                current_state = next_state
                current_action = next_action
                current_steps += 1
                if done or current_steps >= self.max_steps:
                    # if self.use_nn:
                    #     self.critic.batch_update_nn(visited)
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
        "n_episodes": 500,
        "max_steps": 300,
        "use_nn": True,
        "network_dim": (15,30,20,5,1),
        "a_alpha": 0.1,
        "c_alpha": 0.1,
        "gamma": 0.99,
        "lamb": 0.9,
        "epsilon": 0.5,
        "display": 0,
        "frame_delay": 0
    }
    cart_pole_config = {
        "g": 9.81,
        "m_c": 1,
        "m_p": 0.1,
        "l": 0.5,
        "tau": 0.02,
        "buckets": (8,8,16,16)
    }
    world = CartPoleGame(**cart_pole_config)
    system = RLSystem(**global_config, game=world)
    system.train()
    system.visualize_steps()
    # system.visualize_angles()

def test_hanoi() -> None:
    global_config = {
        "n_episodes": 300,
        "max_steps": 40,
        "use_nn": False,
        "network_dim": (100, 50, 25, 30, 1),
        "a_alpha": 0.1,
        "c_alpha": 0.1,
        "gamma": 0.9,
        "lamb": 0.99,
        "epsilon": 0.1,
        "display": 0,
        "frame_delay": 0
    }
    game = Hanoi(n_pegs=6, n_discs=4)
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