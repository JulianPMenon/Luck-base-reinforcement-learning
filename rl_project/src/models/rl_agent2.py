import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, Tuple, List

class RLAgent(nn.Module):
    """DQN agent with intrinsic motivation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim
        
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Copy weights to target network
        self.update_target_network()
        
        # Hyperparameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.intrinsic_weight = 0.1
        
        # Memory
        self.memory = deque(maxlen=10000)
        
    def update_target_network(self):
        """Update target network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, state: torch.Tensor) -> int:
        """Choose action using epsilon-greedy policy"""
        # Debug: print epsilon value
        #print(f"[RLAgent] Epsilon: {self.epsilon:.4f}")
        if random.random() < self.epsilon:
            action = random.randrange(self.action_dim)
            #print(f"[RLAgent] Random action: {action}")
            return action
        with torch.no_grad():
            q_values = self.q_network(state)
            action = q_values.argmax().item()
            #print(f"[RLAgent] Greedy action: {action}, Q-values: {q_values.tolist()}")
            return action
    
    def remember(self, state, action, reward, next_state, done, intrinsic_reward=0):
        """Store experience in memory"""
        total_reward = reward + self.intrinsic_weight * intrinsic_reward
        if reward != 0:
            print(f"[RLAgent] Remember: reward={reward}, intrinsic_reward={intrinsic_reward}, total_reward={total_reward}")
        self.memory.append((state, action, total_reward, next_state, done))
    
    def train(self, batch_size: int = 32) -> float:
        """Train the agent on a batch of experiences"""
        if len(self.memory) < batch_size:
            print("[RLAgent] Not enough memory to train.")
            return 0.0
        batch = random.sample(self.memory, batch_size)
        states = torch.stack([e[0] for e in batch])
        actions = torch.tensor([e[1] for e in batch])
        rewards = torch.tensor([e[2] for e in batch], dtype=torch.float32)
        next_states = torch.stack([e[3] for e in batch])
        dones = torch.tensor([e[4] for e in batch], dtype=torch.bool)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        # Debug: print loss value
        #print(f"[RLAgent] Training loss: {loss.item()}")
        # Backward pass
        optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            #print(f"[RLAgent] Epsilon decayed to: {self.epsilon}")
        return loss.item()