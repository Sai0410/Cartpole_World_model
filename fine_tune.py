import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from collections import deque

base = "/content/drive/MyDrive/Cartpole_world_model"
pretrained = f"{base}/models/policy_net.pt"
finetuned = f"{base}/models/policy_net_finetuned.pt"
plot_path = f"{base}/plots/fine_tuning_rewards.png"


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)


model = PolicyNet()
model.load_state_dict(torch.load(pretrained))
target = PolicyNet()
target.load_state_dict(model.state_dict())
target.eval()

optimizer = optim.Adam(model.parameters(), lr=1e-3)


GAMMA = 0.99
BATCH = 64
EPISODES = 150         
EPSILON = 0.1          
TARGET_UPDATE = 10
REPLAY_CAP = 5000

replay = deque(maxlen=REPLAY_CAP)
env = gym.make("CartPole-v1")   
reward_hist = []

def select_action(state):
    if random.random() < EPSILON:
        return random.randrange(2)
    with torch.no_grad():
        q = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(dim=1).item())

def optimize():
    if len(replay) < BATCH:
        return
    batch = random.sample(replay, BATCH)
    s, a, r, s2, d = zip(*batch)

    s  = torch.tensor(s,  dtype=torch.float32)
    a  = torch.tensor(a,  dtype=torch.int64).unsqueeze(1)
    r  = torch.tensor(r,  dtype=torch.float32).unsqueeze(1)
    s2 = torch.tensor(s2, dtype=torch.float32)
    d  = torch.tensor(d,  dtype=torch.float32).unsqueeze(1)

    q_sa = model(s).gather(1, a)
    with torch.no_grad():
        q_next = target(s2).max(1, keepdim=True)[0]
        y = r + GAMMA * q_next * (1 - d)

    loss = nn.MSELoss()(q_sa, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for ep in range(1, EPISODES+1):
    state, _ = env.reset()
    done = False
    total = 0.0
    while not done:
        action = select_action(state)
        ns, r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay.append((state, action, r, ns, done))
        state = ns
        total += r
        optimize()

    if ep % TARGET_UPDATE == 0:
        target.load_state_dict(model.state_dict())

    reward_hist.append(total)
    print(f"Episode {ep}/{EPISODES} | Reward: {total}")

env.close()


os.makedirs(f"{base}/models", exist_ok=True)
torch.save(model.state_dict(), finetuned)
print(f"Saved fine-tuned model to: {finetuned}")


os.makedirs(f"{base}/plots", exist_ok=True)
plt.figure()
plt.plot(reward_hist)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Fine-tuning reward over episodes")
plt.grid(True)
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved plot to: {plot_path}")


