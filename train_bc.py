
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

class PolicyNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )

  def forward(self, x):
    return self.net(x)


model = PolicyNet()
model.load_state_dict(torch.load('/content/drive/MyDrive/Cartpole_world_model/models/policy_net.pt'))
model.eval()

env = gym.make('CartPole-v1', render_mode='human')

for episode in range(5):
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = model(state_t).argmax(dim=1).item()
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Episode {episode+1}: Total Reward = {total_reward}")

env.close()
