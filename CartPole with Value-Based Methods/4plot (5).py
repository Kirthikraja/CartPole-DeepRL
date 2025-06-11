import numpy as np
import matplotlib.pyplot as plt

#saving model
model_1 = np.load('Naive.npy')  
model_2 = np.load('ER99.npy')  
model_3 = np.load('Tn100.npy')  
model_4 = np.load('Er+Tn.npy')  

default_episodes = 800
x = np.arange(default_episodes)

plt.figure(figsize=(12, 6))
plt.plot(x[:len(model_1)], model_1, label='Naive ', color='blue')  
plt.plot(x[:len(model_2)], model_2, label='Experience Replay(ER) ', color='red')   
plt.plot(x[:len(model_3)], model_3, label='Target Network(TN)', color='green') 
plt.plot(x[:len(model_4)], model_4, label='ER +TN', color='orange')

plt.xlabel('Episodes')  
plt.ylabel('Total Reward per Episode')  
plt.title('Model Comparison')  
plt.legend()  
plt.grid()  

# Show the plot
plt.show()