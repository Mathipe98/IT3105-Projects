import numpy as np


class Actor:

    def __init__(self, alpha: float, gamma: float, lamb: float,
                epsilon: float, n_states: int, n_actions: int) -> None:
        self.alpha, self.gamma, self.lamb = alpha, gamma, lamb
        self.epsilon = epsilon
        self.n_states, self.n_actions = n_states, n_actions
        # Let policy = PI(s), while PI = PI(s,a)
        # policy = 1D array; PI = 2D array
        self.policy = None
        self.PI = None
        self.eligibility_trace = None
        self.debug = 0
        self.initialize()

    def initialize(self) -> None:
        self.PI = np.zeros((self.n_states, self.n_actions),dtype=int)
        self.policy = np.zeros(self.n_states,dtype=int)
        self.eligibility_trace = np.zeros(shape=(self.n_states, self.n_actions))
        self.update_policy_map()
    
    def update_policy_map(self) -> None:
        """Updating PI(s)
        """
        update = lambda state: self.calculate_best_action(state)
        self.policy = np.array(list(map(update, self.policy)))
    
    def calculate_best_action(self, state: int) -> int:
        """Method for calculating the best action given an arbitrary state.
        The method finds all possible actions for the state, and chooses the 
        index that contains the highest value for each action.
        It's also epsilon-greedy, meaning that at any time it can randomly
        choose an action rather than choosing the greedily best one.

        Args:
            state (int): Given state of the world

        Returns:
            int: The action proposed by the policy
        """
        # With probability epsilon, choose a random action
        if np.random.uniform(0,1) < self.epsilon:
            return np.random.choice(range(self.n_actions))
        actions = self.PI[state]
        return np.argmax(actions)
    
    def update(self, delta: float, state: int, action: int) -> None:
        """Updating PI(s, a)

        Args:
            delta (float): [description]
            state (int): [description]
            action (int): [description]
        """
        self.PI[state, action] = self.PI[state, action] + self.alpha * delta
    
    def set_eligibility(self, state: int, action: int=None) -> None:
        self.eligibility_trace[state, action] = 1
    
    def update_eligibility(self, state: int, action: int=None) -> None:
        self.eligibility_trace[state, action] = self.gamma * self.lamb * self.eligibility_trace[state, action]

    def reset_eligibilities(self) -> None:
        self.eligibility_trace = np.zeros(shape=(self.n_states, self.n_actions))
    
    def get_action(self, state) -> int:
        return self.policy[state]

    # def perform_action(self, action: int) -> None:
    #     print(f"State before taking an action:\n\t {self.game.current_state}")
    #     next_state, reward = self.game.get_next_state(action)
    #     print(f"State after taking an action:\n\t {self.game.current_state}")
        


def test_stuff():
    actor = Actor()
    actor.initialize()
    action = actor.calculate_best_action(50)
    for i in range(100):
        print(actor.calculate_best_action(40))

if __name__ == "__main__":
    test_stuff()
