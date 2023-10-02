import numpy as np
import gym

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration-exploitation factor

# Environment setup
env = gym.make('FrozenLake-v1', is_slippery=False)  # Updated to v1
n_actions = env.action_space.n
n_states = env.observation_space.n

# Q-table initialization
Q = np.zeros([n_states, n_actions])

# Training the agent
n_episodes = 10000
for episode in range(n_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[state, :])  # Exploit

        # Take action, observe new state and reward
        next_state, reward, done, _ = env.step(action)

        # Q-learning update rule
        best_next_action = np.argmax(Q[next_state, :])
        target = reward + gamma * Q[next_state, best_next_action]
        Q[state, action] = (1 - alpha) * Q[state, action] + alpha * target

        state = next_state

# Display the trained Q-table
print("Trained Q-table:")
print(Q)

# Testing the agent
total_reward = 0
for episode in range(100):
    state = env.reset()
    done = False
    
    while not done:
        action = np.argmax(Q[state, :])  # Greedy action selection
        state, reward, done, _ = env.step(action)
        total_reward += reward

average_reward = total_reward / 100
print(f"Average Reward Over 100 Episodes: {average_reward:.2f}")

