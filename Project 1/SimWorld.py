from typing import Any, Dict
import numpy as np


class Simworld:

    def __init__(self,
                 game: int,
                 n_episodes: int,
                 n_steps: int,
                 funcapp: bool,
                 network_dim: tuple,
                 act_lr: float,
                 act_decay: float,
                 act_disc: float,
                 crit_lr: float,
                 crit_decay: float,
                 crit_disc: float,
                 epsilon: float,
                 display: Any,  # <- wtf does this do?
                 frame_delay: float) -> None:
        pass

        """
        Thoughts:
            - Create methods for generating (somehow) the general state of the game, some way to represent it
            - Create methods for getting a successor (child) state given a certain action
                * This because the actor will map states to actions, and therefore needs to consult the
                    simworld in order to know what state an action can lead to
            - Create method for receiving an immediate reinforcement (reward) of a state
                * This implies that a mapping must be constructed such that all states have some reinforcement
                * Was going to write something else here, but forgot it
            - Create method for generating the proper start and goal states
            - Create method for checking if a certain state is a goal state
        
        In general, try to come up with a way of representing the different games in similar fashions, such
        that all representations can use the same functionality.
        Maybe coarse coding for all games is possible, look into this.
        """

    def generate_table(self, game: int) -> dict:
        pass
