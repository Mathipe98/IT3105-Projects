import numpy as np
import random
from sim_world import SimWorld
from cart_pole import CartPoleGame

def test_random_survival() -> None:
    config = {
        "game_number": 0,
        "n_episodes": 1,
        "n_steps": 1,
        "funcapp": False,
        "network_dim": None,
        "act_crit_params": [0.1],
        "epsilon": 0,
        "display": 0,
        "frame_delay": 0
    }
    test_features = [-2.3, -0.1, -0.20, -0.1, 0]
    world = SimWorld(**config)
    actions = [-10, 10]
    features = test_features
    state = None
    losing = False
    while not losing:
        action = random.choice(actions)

        features, state = world.get_next_state(features, action)
        losing = world.is_losing_state(state)
        print(f"Features: {features}")
        print(f"State: {state}")
        print(f"Is winning: {world.is_winning_state(state)}")
        print(f"Is losing: {losing}\n")