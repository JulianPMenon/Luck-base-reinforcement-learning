import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, Tuple, List

class RLAgent(nn.Module):
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128, epsilon: float = 1.0, epsilon_decay: float = 0.995, 
                 epsilon_min: float = 0.01, gamma: float = 0.99, learning_rate: float = 1e-3, intrinsic_weight: float = 0.1, batch_size: int = 64, memory_size: int = 1000):
        super().__init__()
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.max_memory = memory_size
        self.memory = deque(maxlen=memory_size)
        self.learning_rate = learning_rate
        self.intrinsic_weight = intrinsic_weight
        self.train_step = 0
        self.target_update_freq = 20
        
        if isinstance(state_size, tuple):
            state_size = int(np.prod(state_size))
        elif state_size is None:
            raise ValueError("state_size cannot be None")
        else:
            state_size = int(state_size)
        self.expected_state_size = state_size
        #print(f"Creating networks with state_size: {state_size}")
                                 
        self.q_network = nn.Sequential(
            nn.Conv2d(state_size, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
        )
        
        self.q_network_head = nn.Sequential(
            nn.Linear(hidden_size*state_size * 3,action_size),
            nn.Softmax()
        )

        self.target_network = nn.Sequential(
            nn.Conv2d(state_size, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
        )

        self.target_network_head = nn.Sequential(
            nn.Linear(hidden_size*state_size * 3,action_size),
            nn.Softmax()
        )
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.update_target_network()
    
    def q_forward(self, x: torch.Tensor):
        features = self.q_network(x)
        z = self.q_network_head(features.view(features.size(0), -1)) # Flatten the features
        return z
    
    def target_forward(self,x: torch.Tensor):
        features = self.target_network(x)
        z = self.target_network_head(features.view(features.size(0), -1)) # Flatten the features
        return z

        
    def update_target_network(self):
        """Copy weights from the Q-network to the target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # Ensure state is a torch tensor
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.q_forward(state.unsqueeze(0))
            return torch.argmax(q_values).item()
        
    def remember(self, state: np.ndarray, action: int,  reward: float, next_state: np.ndarray, done: bool, intrinsic_reward: float = 0.0):
        """Store the experience in memory. Ensures states are torch tensors."""
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)
        if not torch.is_tensor(next_state):
            next_state = torch.tensor(next_state, dtype=torch.float32)
        total_reward = reward + self.intrinsic_weight * intrinsic_reward
        self.memory.append((state, action, total_reward, next_state, done))
        
    def decay_epsilon(self):    
        # Decay epsilon and clamp to minimum
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
            
    def train(self):
        """Train the Q-network using experiences from memory."""
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        # Ensure all states are torch tensors
        states = torch.stack([e[0] if torch.is_tensor(e[0]) else torch.tensor(e[0], dtype=torch.float32) for e in batch])
        actions = torch.tensor([e[1] for e in batch])
        rewards = torch.tensor([e[2] for e in batch], dtype=torch.float32)
        next_states = torch.stack([e[3] if torch.is_tensor(e[3]) else torch.tensor(e[3], dtype=torch.float32) for e in batch])
        dones = torch.tensor([e[4] for e in batch], dtype=torch.float32)
        # Validate state sizes
        if states.shape[1] != self.expected_state_size:
            raise ValueError(f"Training state size mismatch! Expected {self.expected_state_size}, got {states.shape[1]}")
        if next_states.shape[1] != self.expected_state_size:
            raise ValueError(f"Training next_state size mismatch! Expected {self.expected_state_size}, got {next_states.shape[1]}")
        
        current_q_values = self.q_forward(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_forward(next_states).max(1)[0].detach()
        target_q_values = rewards + self.gamma * next_q_values# * (1 - dones)
        criterion = nn.SmoothL1Loss(beta=0.3)
        loss = criterion(current_q_values.squeeze(1), target_q_values)
        # for n in range(len(rewards)):
        #     if rewards[n] != 0:
        #         print(f"reward[n]: {rewards[n]} | current_q_value[n]: {current_q_values[n]} | target_q_values[n]: {target_q_values[n]:.6} | loss {loss:.6}\n")
        # Optimize the Q-network (backwards pass)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()
        return loss.item()