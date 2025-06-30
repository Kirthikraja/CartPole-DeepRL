# CartPole RL



This repository unifies three major classes of deep reinforcement learning algorithms on the CartPole-v1 environment:

- **Value-Based Methods**: DQN with replay buffer & target network  
- **Policy-Gradient Methods**: REINFORCE, Actor–Critic, A2C  
- **Advanced Actor–Critic**: PPO-clipped (and optional SAC)

Each algorithm resides in its own subfolder (`dqn/`, `reinforce/`, `actor_critic/`, `a2c/`, `ppo/`), each containing its own README and training scripts. Shared components—neural network definitions and plotting utilities—are in the root under `networks.py` and `plotting.py`.

## Prerequisites

- Python 3.10  
- Gymnasium 0.27.x  
- PyTorch 2.0.x  
- NumPy 1.23.x  
