import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


epsilon_start = 1.0
epsilon_min = 0.01  
gamma = 0.97  
learning_rate = 0.0000501  
update_to_data_ratio = 2  
network_size = [256, 256, 128]  


epsilon_decays = [0.995, 0.998, 0.999]  


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        layers = []
        input_dim = state_size
        
        for hidden_dim in network_size:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def update_q_network(states, actions, rewards, next_states, dones, q_network, optimizer, device):
    criterion = nn.MSELoss()

    states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device)

    q_predicted = q_network(states_tensor).gather(1, actions_tensor).squeeze(1)

    with torch.no_grad():
        max_next_q = q_network(next_states_tensor).max(dim=1)[0]
        q_target = rewards_tensor + (gamma * max_next_q * (1 - dones_tensor))

    loss = criterion(q_predicted, q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_q_learning(epsilon_decay, num_runs=4):
    num_envs = 8  # Vectorized Environments
    max_steps = 1_000_000  # Total training steps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_returns = []
    
    for run in range(num_runs):
        print(f"\n Training Run {run + 1}/4 for Epsilon Decay: {epsilon_decay}")
        envs = gym.make_vec("CartPole-v1", num_envs=num_envs, vectorization_mode="sync")
        state_size = envs.single_observation_space.shape[0]
        action_size = envs.single_action_space.n  
        q_network = QNetwork(state_size, action_size).to(device)
        optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
        
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
                
                for _ in range(update_to_data_ratio):
                    update_q_network(states, actions, rewards, next_states, dones, q_network, optimizer, device)
                
                states = next_states
                total_rewards += rewards
                total_steps += num_envs  
                done_flags = np.logical_or(done_flags, dones) 
                steps_in_episode += 1  
                if steps_in_episode >= 500:
                    done_flags[:] = True  
                if total_steps >= max_steps:
                    break  
                
            epsilon = max(epsilon_min, epsilon * epsilon_decay)  
            episode_return = np.mean(total_rewards)  
            returns.append(episode_return)

            
            if returns_smooth:
                smoothed_return = smoothing_factor * episode_return + (1 - smoothing_factor) * returns_smooth[-1]
            else:
                smoothed_return = episode_return  
            returns_smooth.append(smoothed_return)
            
            if len(returns) % 10 == 0:
                print(f"Run {run+1}, Episode {len(returns)}, Return: {episode_return:.2f},Smoothed Return: {smoothed_return:.2f}, Total Steps: {total_steps}, Epsilon: {epsilon:.3f}")
            
            if total_steps >= max_steps:
                print(f"✅ Run {run+1} stopped: Reached {max_steps} total steps.")
                break
        
        all_returns.append(returns)
    
    return all_returns


all_results = []
labels = []

for epsilon_decay in epsilon_decays:
    results = train_q_learning(epsilon_decay)
    
   
    max_length = max(len(r) for r in results)

    
    padded_results = [np.pad(r, (0, max_length - len(r)), mode='constant', constant_values=np.nan) for r in results]

    mean_results = np.nanmean(padded_results, axis=0)

    
    window_size = 10
    smoothed_mean_results = np.convolve(mean_results, np.ones(window_size)/window_size, mode='valid')

    
    all_results.append(smoothed_mean_results)
    labels.append(f"Epsilon Decay: {epsilon_decay}")


plt.figure(figsize=(12, 6))
colors = ['b', 'g', 'r']

for i, result in enumerate(all_results):
    smoothed_result = np.convolve(result, np.ones(10)/10, mode='valid')  
    plt.plot(smoothed_result, label=labels[i], color=colors[i], linewidth=2)

plt.xlabel("Number of Episodes")
plt.ylabel("Total Reward per Episode")
plt.title("Ablation Study: Epsilon Decay Comparison")
plt.legend()
plt.grid()
plt.show()

