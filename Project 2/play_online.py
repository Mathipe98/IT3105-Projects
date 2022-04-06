import numpy as np
import os
import pathlib

# Import and initialize your own actor
from mcts import MCTSAgent
from game import Game
from hex import Hex

os.chdir(pathlib.Path(__file__).parent.resolve())

def get_actor() -> MCTSAgent:
    game = Game(game_implementation=Hex(), player=1)
    actor = MCTSAgent(
        game=game,
        tree_traversals=10,
        n_episodes=100,
        model_name="HEX_7x7_FINAL",
    )
    model_params = {
        "hidden_layers": (512, 256,),
        "hl_activations": ('relu', 'sigmoid'),
        "output_activation": 'softmax',
        "optimizer": 'Adam',
        "lr": 0.01,
    }
    actor.setup_model(**model_params)
    actor.train()
    return actor

# Import and override the `handle_get_action` hook in ActorClient
from ActorClient import ActorClient
class MyClient(ActorClient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.series_id = None
        self.flip_gamestate = False

    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        """Called at the start of each set of games against an opponent
        Args:
        unique_id (int): your unique id within the tournament
        series_id (int): whether you are player 1 or player 2
        player_map (list): (inique_id, series_id) touples for both players
        num_games (int): number of games that will be played
        game_params (list): game-specific parameters.
        Note:
        > For the qualifiers, your player_id should always be "-200",
        but this can change later
        > For Hex, game params will be a 1-length list containing
        the size of the game board ([board_size])
        """
        self.logger.info(
        'Series start: unique_id=%s series_id=%s player_map=%s num_games=%s'
        ', game_params=%s',
        unique_id, series_id, player_map, num_games, game_params,
        )
        self.series_id = series_id
        if series_id == 2:
            self.flip_gamestate = True

    def handle_get_action(self, state):
        # Create a numpy array from state and replace 2 values with -1
        row, col = actor.get_tournament_action(state, False)
        return row, col
    
    def handle_game_over(self, winner, end_state):
        """Called after each game
        Args:
        winner (int): the winning player (1 or 2)
        end_stats (tuple): final board configuration
        Note:
        > Given the following end state for a 5x5 Hex game
        state = [
        2, # Current player is 2 (doesn't matter)
        0, 2, 0, 1, 2, # First row
        0, 2, 1, 0, 0, # Second row
        0, 0, 1, 0, 0, # ...
        2, 2, 1, 0, 0,
        0, 1, 0, 0, 0
        ]
        > Player 1 has won here since there is a continuous
        path of ones from the top to the bottom following the
        neighborhood description given in `handle_get_action`
        """
        self.logger.info('Game over: winner=%s', winner)
        print(f"Game over state: {np.array(end_state[1:]).reshape(7,7)}")

# Initialize and run your overridden client when the script is executed
if __name__ == '__main__':
    actor = get_actor()
    client = MyClient(auth="8274151fb5ea436bae5e804d223743c9")
    client.run()