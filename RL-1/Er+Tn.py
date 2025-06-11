
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque


epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995  
learning_rate = 0.0000501  
update_to_data_ratio = 2  
network_size = [256, 256, 128]  
gamma = 0.99  
max_steps = 1_000_000  
target_update_freq = 500 
batch_size = 64  
replay_buffer_size = 50000  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, network_size[0])
        self.fc2 = nn.Linear(network_size[0], network_size[1])
        self.fc3 = nn.Linear(network_size[1], network_size[2])
        self.fc4 = nn.Linear(network_size[2], action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def size(self):
        return len(self.buffer)



def update_q_network(q_network, target_network, optimizer, replay_buffer, batch_size, device):
    if replay_buffer.size() < batch_size:
        return  

    criterion = nn.MSELoss()

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    
    states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device)

    q_predicted = q_network(states_tensor).gather(1, actions_tensor).squeeze(1)

    with torch.no_grad():
        max_next_q = target_network(next_states_tensor).max(dim=1)[0]
        q_target = rewards_tensor + (gamma * max_next_q * (1 - dones_tensor))
    
    loss = criterion(q_predicted, q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_q_learning(num_runs=4):
    num_envs = 8  # Vectorized Environments
    all_returns = []
    all_returns_smooth = []

    for run in range(num_runs):
        print(f"\n Training Run {run + 1}/{num_runs} with TN + ER")
        envs = gym.make_vec("CartPole-v1", num_envs=num_envs, vectorization_mode="sync")
        state_size = envs.single_observation_space.shape[0]
        action_size = envs.single_action_space.n  
        q_network = QNetwork(state_size, action_size).to(device)
        target_network = QNetwork(state_size, action_size).to(device)
        target_network.load_state_dict(q_network.state_dict())  
        optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
        replay_buffer = ReplayBuffer(replay_buffer_size)  
        
        returns = []
        returns_smooth = []
        smoothing_factor = 0.1
        total_steps = 0
        epsilon = epsilon_start  
        
        while total_steps < max_steps:
            states, _ = envs.reset()
            total_rewards = np.zeros(num_envs)
            done_flags = np.zeros(num_envs, dtype=bool)
            steps_in_episode = 0
            
            while not np.all(done_flags):  
                if np.random.rand() < epsilon:
                    actions = np.random.randint(0, action_size, size=num_envs)  
                else:
                    with torch.no_grad():
                        actions = torch.argmax(q_network(torch.tensor(states, dtype=torch.float32).to(device)), dim=1).cpu().numpy()
                
                next_states, rewards, dones, truncations, _ = envs.step(actions)
                
                for i in range(num_envs):
                    replay_buffer.add(states[i], actions[i], rewards[i], next_states[i], dones[i])  # ✅ Store in Replay Buffer

                
                for _ in range(update_to_data_ratio):
                    update_q_network(q_network, target_network, optimizer, replay_buffer, batch_size, device)
                
                states = next_states
                total_rewards += rewards
                total_steps += num_envs  
                done_flags = np.logical_or(done_flags, dones) 
                steps_in_episode += 1  
                if steps_in_episode >= 500:
                    done_flags[:] = True  
                if total_steps >= max_steps:
                    break  
                
                
                if total_steps % target_update_freq == 0:
                    target_network.load_state_dict(q_network.state_dict())
            
            epsilon = max(epsilon_min, epsilon * epsilon_decay)  
            episode_return = np.mean(total_rewards)  
            returns.append(episode_return)

            
            if returns_smooth:
                smoothed_return = smoothing_factor * episode_return + (1 - smoothing_factor) * returns_smooth[-1]
            else:
                smoothed_return = episode_return  
            returns_smooth.append(smoothed_return)

            if len(returns) % 10 == 0:
                print(f"Run {run+1}, Episode {len(returns)}, Return: {episode_return:.2f}, Smoothed Return: {smoothed_return:.2f}, Total Steps: {total_steps}, Epsilon: {epsilon:.3f}")

        all_returns.append(returns)
        all_returns_smooth.append(returns_smooth)

    return all_returns, all_returns_smooth


all_returns, all_returns_smooth = train_q_learning()


plt.figure(figsize=(12, 6))
colors = ['b', 'g', 'r', 'c', 'm']
for i, (run_returns, run_smooth) in enumerate(zip(all_returns, all_returns_smooth)):
    plt.plot(run_smooth, label=f"Run {i+1} (Smoothed)", color=colors[i], linewidth=2)
plt.xlabel("Number of Episodes")
plt.ylabel("Total Reward per Episode")
plt.title("TN + ER: Q-Learning on CartPole")
plt.legend()
plt.grid()
plt.show()



plt.figure(figsize=(12, 6))
max_length = max(len(run) for run in all_returns)


padded_returns = [np.pad(run, (0, max_length - len(run)), 'constant', constant_values=np.nan) for run in all_returns]
mean_returns = np.nanmean(padded_returns, axis=0)


window_size = 10  
smoothed_mean_returns = np.convolve(mean_returns, np.ones(window_size)/window_size, mode='valid')


plt.plot(smoothed_mean_returns, color='black', label="Smoothed Mean Performance")

plt.xlabel("Episodes")
plt.ylabel("Total Reward per Episode")
plt.title("TN + ER: Smoothed Mean Performance")
plt.legend()
plt.grid()
plt.show()

np.save("Er+Tn.npy", smoothed_mean_returns)