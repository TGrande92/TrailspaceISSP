import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import pickle
from collections import namedtuple, deque

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
    model.load_state_dict(torch.load(filename))
    model.eval()  # Set the model to evaluation mode

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

def get_reward(altitude, xaccel, yaccel, zaccel):
    if altitude > 151:
        return 1
    else:
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

def save_commands_to_csv(commands, file_path):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["elapsedTime", "elevator", "aileron", "rudder"])
        writer.writerows(commands)

def run_episode(data):
    """Run one episode and return the total reward."""
    total_reward = 0
    stored_commands = []

    for idx, row in enumerate(data):
        elapsedTime, altitude, xaccel, yaccel, zaccel = row
        altitude = float(altitude)
        xaccel = float(xaccel)
        yaccel = float(yaccel)
        zaccel = float(zaccel)
        state = torch.tensor([altitude, xaccel, yaccel, zaccel], dtype=torch.float32).unsqueeze(0)
        action = select_action(state)
        reward = get_reward(altitude, xaccel, yaccel, zaccel)
        total_reward += reward
        elevator, aileron, rudder = action_space[int(action.item())]
        stored_commands.append([elapsedTime, elevator, aileron, rudder])

        # Store this transition into the replay memory
        if idx + 1 < len(data):
            next_state = torch.tensor([float(data[idx+1][1]), float(data[idx+1][2]), float(data[idx+1][3]), float(data[idx+1][4])], dtype=torch.float32).unsqueeze(0)
        else:
            next_state = None
        memory.push(state, action, next_state, torch.tensor([reward], dtype=torch.float32))

        # End the episode if the drone hits the ground
        if altitude <= 0:
            break

    # Optimize the model after the episode ends
    optimize_model()

    return total_reward, stored_commands

def main():
    data = load_simulator_data("dummydata/data18.csv")
    
    # For this example, we're using the same data for each episode. 
    # In a more realistic scenario, each episode would have different data.
    num_episodes = 10
    for episode in range(num_episodes):
        total_reward, stored_commands = run_episode(data)
        print(f"Episode {episode + 1} Total Reward: {total_reward}")
    
    # Save commands from the last episode for demonstration
    save_commands_to_csv(stored_commands, "output_commands.csv")
    save_replay_memory(memory, "replay_memory.pkl")

    # Save the model weights after all episodes
    save_model_weights(policy_net, "policy_net_weights.pth")
    save_model_weights(target_net, "target_net_weights.pth")

if __name__ == "__main__":
    main()
