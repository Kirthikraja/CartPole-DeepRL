import gymnasium as gym   
import torch              
import torch.nn as nn     
import torch.optim as optim  
import numpy as np        
import random             
import matplotlib.pyplot as plt  
import time
from collections import deque
import random
import time



    

num_0f_envs = 8  
num_runs = 4 

# Enable GPU acceleration
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


AllReturns = []
AllReturnsSmooth = []

# Hyperparameters
EpsilonStart = 1.0  
EpsilonMinimum = 0.01  
EpsilonDecay = 0.998  
AgentLearningRate = 0.0000501  
Gamma = 0.99 #0.99  
MaximumSteps = 1_000_000 

class network(nn.Module):
    def __init__(self, stateSize, actionSize):
        super(network, self).__init__()
        self.fc1 = nn.Linear(stateSize, 256)  
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, actionSize)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)  
    
def update_q_network(state, action, reward, next_states, dones, Qnetwork, optimizer, gamma=0.99
):
    criterion = nn.MSELoss()
    
    states_tensor = torch.tensor(state, dtype=torch.float32).to(Device)
    actions_tensor = torch.tensor(action, dtype=torch.int64).unsqueeze(1).to(Device)
    rewards_tensor = torch.tensor(reward, dtype=torch.float32).to(Device)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(Device)
    dones_tensor = torch.tensor(dones, dtype=torch.float32).to(Device)
    
    q_predicted = Qnetwork(states_tensor).gather(1, actions_tensor).squeeze(1)
    
    with torch.no_grad():
        max_next_q = Qnetwork(next_states_tensor).max(dim=1)[0]
        q_target = rewards_tensor + (gamma * max_next_q * (1 - dones_tensor))
    
    loss = criterion(q_predicted, q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



 

#Training Loop 

for run in range(num_runs):
    print(f"\n Starting the Training Run {run + 1}/{num_runs} on {Device}")
    environment = gym.make_vec("CartPole-v1", num_envs=num_0f_envs, vectorization_mode="sync")
    
    # Reinitialize Q-network and optimizer for each run
    stateSize = environment.single_observation_space.shape[0]
    actionSize = environment.single_action_space.n  
    q_network = network(stateSize, actionSize).to(Device)  
    optimizer = optim.Adam(q_network.parameters(), lr=AgentLearningRate)
    
    smoothingFactor = 0.1  
    TotalSteps = 0  
    epsilon = EpsilonStart 
    returns = []  
    returnsSmooth = []  
    
    
    while TotalSteps < MaximumSteps:
        states, _ = environment.reset()
        total_rewards = np.zeros(num_0f_envs)
        done_flags = np.zeros(num_0f_envs, dtype=bool)
        steps_in_episode = 0
    
        while not np.all(done_flags):  
            if np.random.rand() < epsilon:
                actions = np.random.randint(0, actionSize, size=num_0f_envs)  
            else:
                with torch.no_grad():
                    actions = torch.argmax(q_network(torch.tensor(states, dtype=torch.float32).to(Device)), dim=1).cpu().numpy()
    
            next_states, rewards, dones, truncations, _ = environment.step(actions)
            
            update_q_network(states, actions, rewards, next_states, dones, q_network, optimizer, Gamma)
            
            states = next_states
            total_rewards += rewards
            TotalSteps += num_0f_envs  
            done_flags = np.logical_or(done_flags, dones) 
            steps_in_episode += 1  
    
            if steps_in_episode >= 500:
                done_flags[:] = True  
    
            if TotalSteps >= MaximumSteps:
                break  
    
        epsilon = max(EpsilonMinimum, epsilon * EpsilonDecay)  
        episode_return = np.mean(total_rewards)  
        returns.append(episode_return)  
    
        # smoothing
        if returnsSmooth:
            smoothed_return = smoothingFactor * episode_return + (1 - smoothingFactor) * returnsSmooth[-1]
        else:
            smoothed_return = episode_return  
        returnsSmooth.append(smoothed_return)
    
        if len(returns) % 10 == 0:
            print(f"Run {run + 1}, Episode {len(returns)}, Return: {episode_return:.2f}, Smoothed Return: {smoothed_return:.2f}, Total Steps: {TotalSteps}, Epsilon: {epsilon:.3f}")
        
        if TotalSteps >= MaximumSteps:
            print(f"✅ Run {run + 1} stopped: Reached 1,000,000 total steps.")
            break

    AllReturns.append(returns)
    AllReturnsSmooth.append(returnsSmooth)

# plot
plt.figure(figsize=(12, 6))
colors = ['b', 'g', 'r', 'c', 'm']

for i, (run_returns, run_smooth) in enumerate(zip(AllReturns, AllReturnsSmooth)):
    # plt.plot(run_returns, label=f"Run {i+1} (Raw)", color=colors[i], alpha=0.4)
    plt.plot(run_smooth, label=f"Run {i+1} (Smoothed)", color=colors[i], linewidth=2)

plt.xlabel("Number of Episodes")
plt.ylabel("Total Reward per Episode")
plt.title(" Naive Q-Learning on CartPole")
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
max_length = max(len(run) for run in AllReturns)


padded_returns = [np.pad(run, (0, max_length - len(run)), 'constant', constant_values=np.nan) for run in AllReturns]
mean_returns = np.nanmean(padded_returns, axis=0)

# Applyed simple moving average for smoothing
window_size = 10  # Adjust this for more or less smoothing
smoothed_mean_returns = np.convolve(mean_returns, np.ones(window_size)/window_size, mode='valid')

# Plot the smoothed mean line
plt.plot(smoothed_mean_returns, color='black', label="Smoothed Mean Performance")

plt.xlabel("Episodes")
plt.ylabel("Total Reward per Episode")
plt.title("Naive Q-Learning on CartPole")
plt.legend()
plt.grid()

plt.show()

np.save("Naive.npy", smoothed_mean_returns)