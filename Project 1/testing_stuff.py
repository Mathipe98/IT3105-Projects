import numpy as np
import random
# from sim_world import SimWorld
from cart_pole import CartPoleGame

def test_random_survival() -> None:
    world = CartPoleGame()
    world.reset()
    actions = [0, 1]
    state = world.current_state
    losing = False
    while not losing:
        action = random.choice(actions)
        print(f"Current action: {10 - 20 * action}")
        print(f"Current state parameters: {world.current_state_parameters}")
        state, _ = world.step(action)
        losing = world.is_losing_state(state)
        print(f"Next state parameters: {world.current_state_parameters}")
        print(f"State: {state}")
        print(f"Is winning: {world.is_winning_state(state)}")
        print(f"Is losing: {losing}\n")

def test_oscillation() -> None:
    world = CartPoleGame()
    world.reset()
    # world.theta[0] = 0
    state = world.current_state
    losing = False
    counter = 0
    action = 0
    while not losing:
        if counter == 0:
            action = 0
            counter += 1
        else:
            action = 1
            counter = 0
        print(f"Current action: {10 - 20 * action}")
        print(f"Current state parameters: {world.current_state_parameters}")
        state, _ = world.step(action)
        losing = world.is_losing_state(state)
        print(f"Next state parameters: {world.current_state_parameters}")
        print(f"State: {state}")
        print(f"Is winning: {world.is_winning_state(state)}")
        print(f"Is losing: {losing}\n")
    print(f"Starting theta: {world.theta[0]}")

if __name__ == '__main__':
    test_oscillation()