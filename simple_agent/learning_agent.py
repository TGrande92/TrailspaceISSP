#Andrew Hull
#Monday, Oct 2
#Discrete state ex.
import numpy as np

class QLearningAgent:
    def __init__(self, n_actions, n_states, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_actions = n_actions  # number of actions
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration-exploitation factor
        self.Q = np.zeros((n_states, n_actions))  # initialize Q-table with zeros

    def choose_action(self, state):
        """
        Choose action using epsilon-greedy policy.
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_actions)  # explore: randomly choose an action
        else:
            return np.argmax(self.Q[state, :])  # exploit: choose the action with max Q value for the current state

    def learn(self, state, action, reward, next_state):
        """
        Update the Q-values using the Q-learning update rule.
        """
        predict = self.Q[state, action]
        target = reward + self.gamma * np.max(self.Q[next_state, :])
        self.Q[state, action] = predict + self.alpha * (target - predict)

# Example usage:
# n_actions = 4  # number of actions in the environment
# n_states = 1000  # number of states in the environment
# agent = QLearningAgent(n_actions, n_states)

