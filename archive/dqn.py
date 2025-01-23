import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import os

# Keep the ReplayMemory class the same
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Modified DQN for our state space
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class LunarAgent:
    def __init__(self, env, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.env = env
        self.device = device
        
        # Hyperparameters
        self.batch_size = 128
        self.gamma = 0.99
        self.eps_start = 0.95
        self.eps_end = 0.05
        self.eps_decay = 20000
        self.target_update = 10
        self.memory_size = 50000
        self.lr = 1e-4
        
        # State size calculation (all numerical values from observation)
        self.state_size = 8  # height, battery, pos(2), sunlight, dust, water_prob, gold_prob
        
        # Networks
        self.policy_net = DQN(self.state_size, 64, env.action_space.n).to(device)
        self.target_net = DQN(self.state_size, 64, env.action_space.n).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayMemory(self.memory_size)
        
        # Save paths
        self.checkpoint_dir = './checkpoints'
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
    def process_state(self, state):
        """Convert state list into flat array for neural network"""
        height, battery, pos, sunlight, dust, water_prob, gold_prob = state
        
        # Normalize all values to similar ranges
        norm_height = height / 5.0  # [-5,5] -> [-1,1]
        norm_battery = battery / 100.0  # [0,100] -> [0,1]
        norm_pos_x = pos[0] / (self.env.grid_height - 1)  # [0,34] -> [0,1]
        norm_pos_y = pos[1] / (self.env.grid_width - 1)   # [0,34] -> [0,1]
        # sunlight already [0,1]
        # dust already [0,0.5]
        # probabilities already [0,1]
        
        return np.array([
            norm_height,
            norm_battery,
            norm_pos_x,
            norm_pos_y,
            sunlight,
            dust,
            water_prob,
            gold_prob
        ], dtype=np.float32)

    def select_action(self, state, steps_done):
        """Select action using epsilon-greedy policy"""
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * steps_done / self.eps_decay)
            
        if sample > eps_threshold:
            with torch.no_grad():
                # Process state and pass through network
                state_tensor = torch.FloatTensor(self.process_state(state)).unsqueeze(0).to(self.device)
                return self.policy_net(state_tensor).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.env.action_space.n)]], 
                            device=self.device, dtype=torch.long)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None
            
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                    device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([torch.FloatTensor(self.process_state(s)).unsqueeze(0) 
                                         for s in batch.next_state if s is not None]).to(self.device)
        
        state_batch = torch.cat([torch.FloatTensor(self.process_state(s)).unsqueeze(0) 
                               for s in batch.state]).to(self.device)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes, max_steps=500):
        episode_rewards = []
        steps_done = 0
        
        for i_episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            losses = []
            
            for t in range(max_steps):
                action = self.select_action(state, steps_done)
                steps_done += 1
                
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                total_reward += reward
                
                reward = torch.tensor([reward], device=self.device)
                
                if terminated:
                    next_state = None
                
                self.memory.push(state, action, next_state, reward)
                state = next_state
                
                loss = self.optimize_model()
                if loss is not None:
                    losses.append(loss)
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(total_reward)
            
            # Print progress
            if i_episode % 1000 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {i_episode}")
                print(f"Average Reward (last 10): {avg_reward:.2f}")
                print(f"Steps Done: {steps_done}")
                
            # Update target network
            if i_episode % self.target_update == 50:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
            # Save checkpoint
            if i_episode % 1000 == 0:
                torch.save({
                    'episode': i_episode,
                    'model_state_dict': self.policy_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'episode_rewards': episode_rewards,
                }, f"{self.checkpoint_dir}/checkpoint_episode_{i_episode}.pt")
        
        return episode_rewards