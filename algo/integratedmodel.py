import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import pickle
from collections import namedtuple, deque
from fgclient import FgClient
import time
import subprocess
import os
import glob

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
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
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
memory = ReplayMemory(10000)
steps_done = 0

# Select Action Function
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)

def get_reward(altitude, flight_duration):
    if altitude > 153:
        # Positive reward for staying above the crash threshold
        # Increase the reward linearly with flight duration
        return 1 + flight_duration * 0.1
    else:
        # Large negative reward for crashing
        return -100

# Optimize Model Function
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
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
        try:
            print(f"Starting Flightgear for Run No {run_number}!")
            run_fgfs('run_fg_in.sh')
            time.sleep(10)
            client = FgClient()

            start_time = time.time()
            episode_actions = []
            episode_states = []
            episode_actions_log = []
            while client.altitude_ft() > 153:
                # Get the current state
                altitude = client.altitude_ft()
                xaccel = client.get_xaccel()
                yaccel = client.get_yaccel()
                zaccel = client.get_zaccel()
                states = [altitude, xaccel, yaccel, zaccel]
                episode_states.append(states)
                print(altitude, xaccel, yaccel, zaccel)

                state = torch.tensor([altitude, xaccel, yaccel, zaccel], dtype=torch.float32).unsqueeze(0)

                # Process state to get actions
                action_index = select_action(state)
                elevator, aileron, rudder = action_space[int(action_index.item())]
                actions = [elevator, aileron, rudder]
                episode_actions_log.append(actions)
                print(elevator, aileron, rudder)

                # Apply actions to the simulator
                client.set_elevator(elevator)
                client.set_aileron(aileron)
                client.set_rudder(rudder)

                # Store the state and action
                episode_actions.append((state, action_index))

            run_end_time = time.time()
            flight_duration = run_end_time - start_time
            print(f"Flight duration for Run No {run_number}: {flight_duration} seconds")

            # Update memory and optimize model with the episode reward
            final_altitude = client.altitude_ft()
            reward = get_reward(final_altitude, flight_duration)
            log_run_data(current_run_log, episode_states, episode_actions_log, flight_duration, reward)
            print(f'Run reward: {reward}')
            for i in range(len(episode_actions)):
                state, action_index = episode_actions[i]
                action_tensor = torch.tensor([[action_index]], dtype=torch.long)
                
                if i < len(episode_actions) - 1:
                    next_state = episode_actions[i + 1][0]  # Next state is the state of the next action
                else:
                    next_state = None  # No next state for the last action

                update_memory(state, action_tensor, next_state, reward)

            run_number += 1
            current_run_log += 1
            if run_number % 5 == 0:
                optimize_model()

        finally:
            kill_fgfs()

    # Save model and memory at the end
    save_model_weights(policy_net, "policy_net_weights.pth")
    save_model_weights(target_net, "target_net_weights.pth")
    save_replay_memory(memory, "replay_memory.pkl")

if __name__ == "__main__":
    main()
