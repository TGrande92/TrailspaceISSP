import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class SimpleDroneEnv(gym.Env):
    def __init__(self):
        super(SimpleDroneEnv, self).__init__()

        # Define action space: 4 discrete actions (up, down, left, right)
        self.action_space = spaces.Discrete(4)

        # Define observation space: drone position (x, y), battery level
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([10, 10, 100]), dtype=np.float32)

        # Initial state
        self.state = np.array([5, 5, 100])  # drone starts at position (5, 5) with full battery

    def step(self, action):
        # Apply action (update drone's position)
        if action == 0:   # up
            self.state[1] += 1
        elif action == 1: # down
            self.state[1] -= 1
        elif action == 2: # left
            self.state[0] -= 1
        elif action == 3: # right
            self.state[0] += 1

        # Decrease battery level
        self.state[2] -= 1

        # Calculate reward (simplified for this example)
        reward = self.state[2]

        # Check if episode is done (battery is empty or drone is out of bounds)
        done = self.state[2] <= 0 or \
               self.state[0] < 0 or self.state[0] > 10 or \
               self.state[1] < 0 or self.state[1] > 10

        return self.state, reward, done, {}

    def reset(self):
        # Reset state to initial condition for new episode
        self.state = np.array([5, 5, 100])
        return self.state

    def render(self, mode='human'):
        plt.clf()  # clear the current figure
        plt.xlim(0, 10)  # set the x-limits of the current axes
        plt.ylim(0, 10)  # set the y-limits of the current axes
        plt.plot(self.state[0], self.state[1], 'bo')  # plot the drone's position as a blue dot
        plt.pause(0.1)  # pause for a short duration before the next render

