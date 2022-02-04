import numpy as np
from cart_pole import CartPoleGame
from actor import Actor

class Critic:

    def __init__(self, sim_world: CartPoleGame, use_nn: bool=False, actor: Actor=None) -> None:
        self.sim_world = sim_world
        self.use_nn = use_nn
        self.n_states = self.sim_world.n_states
        self.n_actions = self.sim_world.n_actions
        # If we're using neural network, then state eval regards only state
        # If we're using SAP, then the value must be tied to both state and action
        self.state_eval = None
        self.actor = actor
    
    def initialize(self) -> None:
        if self.use_nn:
            self.state_eval = np.random.uniform(low=-0.1, high=0.1, size=(self.n_states,))
        else:
            self.state_eval = np.random.uniform(low=-0.1, high=0.1, size=(self.n_states,self.n_actions))
    

    def calculate_values(self):
        
        pass

def test_stuff():
    world = CartPoleGame()
    critic = Critic(world)
    critic.initialize()
    print()

if __name__ == "__main__":
    test_stuff()