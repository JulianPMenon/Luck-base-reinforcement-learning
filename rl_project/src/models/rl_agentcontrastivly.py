import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, Tuple, List

class RLAgent(nn.Module):
    """DQN agent with intrinsic motivation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, epsilon_decay:float = 0.995, gamma = 0.999):
        super().__init__()
        self.action_dim = action_dim
        
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            #nn.Sigmoid()
        )
        
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            #nn.Sigmoid()
        )
        
        # Copy weights to target network
        self.update_target_network()
        
        # Hyperparameters
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.gamma = gamma
        self.intrinsic_weight = 0.1
        
        # Memory
        self.memory = deque(maxlen=1000)
        
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
            #print(f"state: {state.unsqueeze(0).shape}")
            q_values = self.q_network(state.unsqueeze(0)/32)
            action = q_values.argmax().item()
            #print(f"[RLAgent] Greedy action: {action}, Q-values: {q_values.tolist()}")
            return action
    
    def remember(self, state, action, reward, next_state, done, intrinsic_reward=0):
        """Store experience in memory"""
        total_reward = reward + self.intrinsic_weight * intrinsic_reward
        # if reward != 0:
        # print(f"[RLAgent] Remember: reward={reward}, intrinsic_reward={intrinsic_reward}, total_reward={total_reward}")
        self.memory.append((state, action, total_reward, next_state, done))
    
    def train(self, optimizer, batch_size: int = 32) -> float:
        """Train the agent on a batch of experiences"""
        if len(self.memory) < batch_size:
            print("[RLAgent] Not enough memory to train.")
            return 0.0
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        batch = random.sample(self.memory, batch_size)
        states = torch.stack([e[0] for e in batch]).to(device)
        actions = torch.tensor([e[1] for e in batch]).to(device)
        rewards = torch.tensor([e[2] for e in batch], dtype=torch.float32).to(device)
        next_states = torch.stack([e[3] for e in batch]).to(device)
        dones = torch.tensor([e[4] for e in batch], dtype=torch.bool).to(device)
            
        #print(f"states: {states.shape}")
        current_q_values = self.q_network(states/batch_size)
        if float('nan') in current_q_values[0]:
            print(f"current_state:    {states}")
            print(f"current_q_values: {current_q_values}")
        #print(f"current_q_values {current_q_values.shape}, actions.unsqueeze(1) {actions.unsqueeze(1).shape}")
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_network(next_states/batch_size).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        criterion = nn.L1Loss()
        loss = criterion(current_q_values.squeeze(), target_q_values)
        # Debug: print loss value
        #print(f"[RLAgent] Training loss: {loss.item()}")
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Decay epsilon
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
            #print(f"[RLAgent] Epsilon decayed to: {self.epsilon}")
        return loss.item()

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
