#Andrew Hull
#Monday, Oct 2
#Discrete state ex.
import matplotlib.pyplot as plt
from simple_agent import SimpleDroneEnv 
from learning_agent import QLearningAgent 

# Instantiate environment and agent
env = SimpleDroneEnv()
n_actions = env.action_space.n
n_states = 1000 

agent = QLearningAgent(n_actions=n_actions, n_states=n_states)

# Set number of episodes
n_episodes = 1000

# List to store cumulative reward of each episode
rewards = []

# Helper function to discretize states
def discretize_state(state):
    # Adjust the calculation logic to fit your specific state characteristics
    discrete_state = int(state[0]*100 + state[1]*10 + state[2])
    # Ensure discrete_state is within valid bounds
    discrete_state = max(0, min(discrete_state, 999))  # assuming Q-table size is 1000
    return discrete_state


# Training loop
for episode in range(n_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Discretize state
        discrete_state = discretize_state(state)
        
        action = agent.choose_action(discrete_state)
        next_state, reward, done, _ = env.step(action)
        
        # Discretize next_state
        discrete_next_state = discretize_state(next_state)
        
        # Learning step
        agent.learn(discrete_state, action, reward, discrete_next_state)
        
        state = next_state
        total_reward += reward
        
    rewards.append(total_reward)

    # Optional: Render every episode or every few episodes
    if episode % 10 == 0:
        env.render()

# Plotting the learning curve
plt.figure(figsize=(10,5))
plt.plot(rewards)
plt.title('Learning Curve')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
