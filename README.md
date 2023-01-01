# IT3105-Projects
This is a repository containing code for 2 of the projects in the course IT3105 - Artifical Intelligence Programming at NTNU, Spring 2022.

## Project 1

The first project concerns the actor-critic model, in which an actor is the entity whose responsibility is choosing an action in the world, and the critic 
is responsible for telling the actor how good the chosen action was.

In this case, the paradigm was applied to the Cartpole-problem, i.e. having a movable bar/pendulum attached on top of a cart pole, and creating a system that
teaches itself the appropriate actions in order to balance it.

The results can be seen by simply running the initialize.py file, where the model will be trained, and the model will (hopefully) eventually converge
to a stable solution where it is able to maintain the pole ontop of itself.

# Project 2

The second project deals with Monte-Carlo Tree-Search, i.e. the same algorithm used by Google when developing AlphaGo/AlphaZero, which are state-of-the-art 
Go and Chess-playing AI models respectively.

This algorithm was used to train a system to play the board-game Hex, where the goal is to connect a "bridge" between two opposite sides of the board before the
opponent does so. The models were trained purely by using this algorithm, and in the latter stages of the course, we played online tournaments with our
models in order to see whose model would play the best games.

You can compete against the model by simply running the initialize.py file, making sure that the configuration variable "play_network" is equal to true. This
lets you compete directly against the top-trained model in the game through a visual interface (disclaimer: the model is not very good due to not being
inserted with expert knowledge, i.e. some metric that indirectly hints the model of how good its playing).

Note: the credential files needed for online play have been removed, and the models are therefore only available locally.
