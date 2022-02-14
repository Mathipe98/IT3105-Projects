import numpy as np
from collections import deque
import numpy as np
from pyglet.window import key
import time

bool_quit = False
import gym


#Building the environment
env = gym.make('CartPole-v0')


def run_cartPole(policy, n_episodes=1000, max_t=1000, print_every=100, render_env=True, record_video=False):
    """Run the CartPole-v0 environment.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        print_every (int): how often to print average score (over last 100 episodes)

    Adapted from:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/hill-climbing/Hill_Climbing.ipynb

    """
    global bool_quit

    if policy.__class__.__name__ == 'Policy_Human':
        global action
        action = 0  # Global variable used for manual control with key_press
        env = CartPoleEnv()  # This is mandatory for keyboard input
        env.reset()  # This is mandatory for keyboard input
        env.render()  # This is mandatory for keyboard input
        env.viewer.window.on_key_press = key_press  # Quit properly & human keyboard inputs
    else:
        print('** Evaluating', policy.__class__.__name__, '**')
        # Define the Environments
        env = gym.make('CartPole-v0')
        # Set random generator for reproductible runs
        env.seed(0)
        np.random.seed(0)

    if record_video:
        env.monitor.start('/tmp/video-test', force=True)

    scores_deque = deque(maxlen=100)
    scores = []
    trials_to_solve=[]

    for i_episode in range(1, n_episodes+1):
        rewards = []
        state = env.reset()
        if 'reset' in dir(policy):  # Check if the .reset method exists
            policy.reset(state)
        for t in range(max_t):  # Avoid stucked episodes
            action = policy.act(state)
            state, reward, done, info = env.step(action)
            rewards.append(reward)
            if 'memorize' in dir(policy):  # Check if the .memorize method exists
                policy.memorize(state, action, reward, done)
            if render_env: # Faster, but you can as well call env.render() every time to play full window.
                env.render()  # (mode='rgb_array')  # Added  # Changed mode

            if done:  # if Pole Angle is more than +-12 deg or Cart Position is more than +-2.4 (center of the cart reaches the edge of the display) the simulation ends
                trials_to_solve.append(t)
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        if 'update' in dir(policy):  # Check if the .update method exists
            policy.update(state)  # Update the policy

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}\tSteps: {:d}'.format(i_episode, np.mean(scores_deque), t))
        if np.mean(scores_deque) >= 195.0:
            print('Episode {}\tAverage Score: {:.2f}\tSteps: {:d}'.format(i_episode, np.mean(scores_deque), t))
            print('** Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(max(1, i_episode-100), np.mean(scores_deque)))
            break
        if bool_quit:
            break

    if np.mean(scores_deque) < 195.0:
        print('** The environment has never been solved!')
        print('   Mean scores on all runs was < 195.0')
    if record_video:
        env.env.close()
    env.close()
    return scores, trials_to_solve

# Original code from https://ferdinand-muetsch.de/cartpole-with-qlearning-first-experiences-with-openai-gym.html
import math, random
# Ferdinand Mütsch ran a GridSearch on hyperparameters:
#   Best hyperparameters: 'buckets': (1, 1, 6, 12), 'min_alpha': 0.1, 'min_epsilon': 0.1

# Define a Q-Learning Policy
class Policy_QLearning():
    def __init__(self, buckets=(1, 1, 6, 12,), min_alpha=0.1, min_epsilon=0.1, gamma=1.0, ada_divisor=25, state_space_dim=4, action_space_dim=2):
        self.buckets = buckets # down-scaling feature space to discrete range
        self.min_alpha = min_alpha # Learning Rate
        self.min_epsilon = min_epsilon # Exploration Rate: # to avoid local-minima, instead of the best action, chance ε of picking a random action
        self.gamma = gamma # Discount Rate factor
        self.ada_divisor = ada_divisor # only for development purposes

        self.action_space_dim = action_space_dim
        self.Q = np.zeros(self.buckets + (self.action_space_dim,))
        self.rewards = [0]  # Initialize to 0
        self.action = 0  # Initialize to 0

    def discretize(self, obs):
        # Ferdinand choose to
        upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
        lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def act(self, state):
        new_state = self.discretize(state)
        self.update_q(self.current_state, self.action, self.rewards[-1], new_state, self.alpha)
        self.current_state = new_state

        if np.random.rand() <= self.epsilon:
            self.action = random.randrange(self.action_space_dim)
            return self.action
        self.action = np.argmax(self.Q[self.current_state])
        return self.action

    def memorize(self, next_state, action, reward, done):
        self.rewards.append(reward)

    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q[state_old][action] += self.alpha * '** Update the Q matrix here **'

    def reset(self, state):
        # Decrease alpha and epsilon while experimenting
        t = len(self.rewards)
        self.alpha = max(self.min_alpha, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))
        self.epsilon = max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))
        self.current_state = self.discretize(state)

policy = Policy_QLearning()
scores, trials_to_solve = run_cartPole(policy, n_episodes=1000, print_every=100, render_env=False)
print('** Mean average score:', np.mean(scores))
plot_performance(scores)
plot_trials_to_solve(trials_to_solve)