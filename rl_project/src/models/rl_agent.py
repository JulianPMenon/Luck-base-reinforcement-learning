import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, Tuple, List

class RLAgent:
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256, epsilon: float = 1.0, epsilon_decay: float = 0.995, 
                 epsilon_min: float = 0.01, gamma: float = 0.99, learning_rate: float = 1e-3, intrinsic_weight: float = 0.1, batch_size: int = 64, memory_size: int = 10000):
        super().__init__()
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
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
        print(f"Creating networks with state_size: {state_size}")
                                 
        self.q_network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        self.target_network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.update_target_network()
        
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
            if state.dim() > 1:
                state = state.flatten()
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if state.shape[1] != self.expected_state_size:
                raise ValueError(f"State size mismatch! Expected {self.expected_state_size}, got {state.shape[1]}")
            q_values = self.q_network(state)
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
        states = torch.stack([e[0].flatten() if torch.is_tensor(e[0]) else torch.tensor(e[0], dtype=torch.float32).flatten() for e in batch])
        actions = torch.tensor([e[1] for e in batch])
        rewards = torch.tensor([e[2] for e in batch], dtype=torch.float32)
        next_states = torch.stack([e[3].flatten() if torch.is_tensor(e[3]) else torch.tensor(e[3], dtype=torch.float32).flatten() for e in batch])
        dones = torch.tensor([e[4] for e in batch], dtype=torch.float32)
        # Validate state sizes
        if states.shape[1] != self.expected_state_size:
            raise ValueError(f"Training state size mismatch! Expected {self.expected_state_size}, got {states.shape[1]}")
        if next_states.shape[1] != self.expected_state_size:
            raise ValueError(f"Training next_state size mismatch! Expected {self.expected_state_size}, got {next_states.shape[1]}")
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(current_q_values.squeeze(1), target_q_values)
        # Optimize the Q-network (backwards pass)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()
        return loss.item()