import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import pickle
from collections import namedtuple, deque
import JSBSim
import time
import subprocess
import os
import glob
import numpy as np
import statistics

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# DQN Network
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def save_replay_memory(memory, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(memory, file)

def load_replay_memory(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

# Function to save model weights
def save_model_weights(model, filename):
    torch.save(model.state_dict(), filename)

# Function to load model weights
def load_model_weights(model, filename):
    if os.path.exists(filename):
        model.load_state_dict(torch.load(filename))
        model.train()  # Set the model to training mode
    else:
        print(f"No weights found at {filename}, initializing model with default weights.")
        torch.save(model.state_dict(), filename)
        model.train()

# Global Variables
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 2000
TAU = 0.005
LR = 1e-4

# action space with 0.1 intervals
action_space = [
    (elevator, aileron, rudder)
    for elevator in [i/10 for i in range(-10, 11)]
    for aileron in [i/10 for i in range(-10, 11)]
    for rudder in [i/10 for i in range(-10, 11)]
]

n_actions = len(action_space)
n_observations = 4  # altitude, xaccel, yaccel, zaccel

policy_net = DQN(n_observations, n_actions)
target_net = DQN(n_observations, n_actions)
load_model_weights(policy_net, "policy_net_weights.pth")
load_model_weights(target_net, "target_net_weights.pth")
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(1000000)
steps_done = 0

# Function to dynamically adjust epsilon
def dynamic_eps(run_number, steps_done, total_runs=10, min_eps=0.01, max_eps=1.0):
    return min_eps + (max_eps - min_eps) * math.exp(-1. * steps_done / EPS_DECAY)

# Select Action Function with Dynamic Epsilon
def select_action(state, run_number, steps_done):
    eps_threshold = dynamic_eps(run_number, steps_done)
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)

def get_state_reward(current_state, altitude_change):
    current_altitude = current_state[0]

    steep_decline_rate = -10  # Example rate in feet per 0.5 second

    reward = 0

    # Penalty for steep decline
    if altitude_change < steep_decline_rate * 0.5:  # Assuming a 0.5 second interval
        reward -= 5

    # Penalty for dropping altitude to zero or below
    if current_altitude <= 0:
        reward -= 10

    return reward

def calculate_stability_reward(altitudes, threshold=5, stable_range=10):
    """
    Calculate stability reward based on altitude variations over the last 'threshold' seconds.
    'stable_range' defines the acceptable range of altitude variation for stability.
    """
    if len(altitudes) < threshold:
        return 0  # Not enough data to calculate stability

    recent_altitudes = altitudes[-threshold:]
    altitude_variance = np.var(recent_altitudes)

    if altitude_variance <= stable_range**2:
        return 2  # Reward for maintaining stable altitude
    else:
        return 0  # No reward if altitude varies too much

def get_episodic_reward(flight_duration, episodes_for_average, directory):
    # Reward based on flight duration
    # Episodic reward is equal to deviation from average flight time of previous series of episodes 

    all_files = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    sorted_files = sorted(all_files, key=os.path.getmtime, reverse=True)

    # Get the last n files
    last_n_files = sorted_files[:episodes_for_average]

    # Read the last line of each file
    last_lines = []
    for file in last_n_files:
        with open(file, 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                last_line = last_line.split(",,,,")
                last_line = last_line[-1]
                last_line = float(last_line)
                last_lines.append(last_line)

    return flight_duration - statistics.mean(last_lines)

# Optimize Model Function
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) if any(non_final_mask) else None
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE)
    if non_final_next_states is not None:
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def load_simulator_data(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        data = [line for line in reader]
    return data

def log_run_data(run_number, episode_states, episode_actions, flight_duration, reward):
    """Logs the data for a single run to a CSV file."""
    run_log_file = f'runlogs/run{run_number}.csv'
    with open(run_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["altitude", "xaccel", "yaccel", "zaccel", "elevator", "aileron", "rudder"])
        for state, action in zip(episode_states, episode_actions):
            writer.writerow(state + list(action))
        writer.writerow(["Reward", "", "", "", reward])
        writer.writerow(["Flight Duration", "", "", "", flight_duration])

def initialize_runlogs():
    """Initializes the runlogs directory and finds the last run number."""
    if not os.path.exists('runlogs'):
        os.makedirs('runlogs')
    
    run_files = glob.glob('runlogs/run*.csv')
    run_numbers = [int(os.path.splitext(os.path.basename(f))[0][3:]) for f in run_files]
    return max(run_numbers) if run_numbers else 0

def process_state(altitude, xaccel, yaccel, zaccel):
    """Process a single state and return the selected action."""
    state = torch.tensor([altitude, xaccel, yaccel, zaccel], dtype=torch.float32).unsqueeze(0)
    action = select_action(state)
    elevator, aileron, rudder = action_space[int(action.item())]
    return (elevator, aileron, rudder)

def update_memory(state, action, next_state, reward):
    """Store the transition in replay memory"""
    memory.push(state, action, next_state, torch.tensor([reward], dtype=torch.float32))
    

def run_fgfs(filename):
    with open(filename, 'r') as file:
        fg_command = file.read().strip()  # Read and strip any trailing whitespace
    return subprocess.Popen(fg_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def kill_fgfs():
    subprocess.run(["bash", "kill.sh"])

def main():
    last_run_number = initialize_runlogs()
    current_run_log = last_run_number + 1
    run_number = 0

    while run_number < 11:
        jsbsim_client = JSBSim.JsbsimInterface()
        steps_done = 0  # Reset steps_done for each run
        altitudes = []  # List to store altitudes for stability calculation
        stability_reward_interval = 100  # Assuming simulation step is 0.01s, calculate every second
        stability_reward = 0

        try:
            print(f"Starting JSBSim for Run No {run_number}!")
            jsbsim_client.start()

            episode_actions = []
            episode_states = []
            episode_actions_log = []
            last_altitude = None

            while True:
                jsbsim_client.run()

                current_state = jsbsim_client.get_state()
                altitude, xaccel, yaccel, zaccel = current_state
                episode_states.append(current_state)

                # Calculate altitude change
                altitude_change = 0
                if last_altitude is not None:
                    altitude_change = altitude - last_altitude
                last_altitude = altitude

                altitudes.append(altitude)

                if len(altitudes) % stability_reward_interval == 0:
                    stability_reward = calculate_stability_reward(altitudes, threshold=100, stable_range=10)

                if jsbsim_client.gear_contact():
                    print("Landing gear contact. Ending simulation.")
                    next_state = None  # Set next state to None if simulation ends
                    break

                if altitude <= 0:  # Check for simulation end condition
                    next_state = None
                    break

                # Process state to get actions
                torch_current_state = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)
                action_index = select_action(torch_current_state, run_number, steps_done)
                elevator, aileron, rudder = action_space[int(action_index.item())]
                actions = [elevator, aileron, rudder]
                episode_actions_log.append(actions)

                # Apply actions to JSBSim
                jsbsim_client.set_controls(elevator, aileron, rudder)

                # Get the next state
                jsbsim_client.run()
                next_state = jsbsim_client.get_state()

                # Store the state and action
                torch_next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) if next_state is not None else None
                episode_actions.append((torch_current_state, action_index))

                # Calculate and update memory with state reward
                state_reward = get_state_reward(current_state, altitude_change)
                # Update the total reward including stability_reward
                total_reward = state_reward + stability_reward
                # print(total_reward) uncomment this to see the total_reward of each state
                action_tensor = torch.tensor([[action_index]], dtype=torch.long)
                update_memory(torch_current_state, action_tensor, torch_next_state, total_reward)

            # Use JSBSim's internal simulation time as flight duration
            flight_duration = jsbsim_client.get_sim_time()
            print(f"Flight duration for Run No {run_number}: {flight_duration} seconds")

            # Calculate episodic reward and update memory
            episodic_reward = get_episodic_reward(flight_duration, 100, "runlogs")
            for state, action in episode_actions:
                update_memory(state, action, None, episodic_reward)

            # Log run data
            log_run_data(current_run_log, episode_states, episode_actions_log, flight_duration, episodic_reward)

            run_number += 1
            current_run_log += 1
            if run_number % 5 == 0:
                optimize_model()

        finally:
            jsbsim_client.stop()

    save_model_weights(policy_net, "policy_net_weights.pth")
    save_model_weights(target_net, "target_net_weights.pth")
    save_replay_memory(memory, "replay_memory.pkl")

if __name__ == "__main__":
    main()

