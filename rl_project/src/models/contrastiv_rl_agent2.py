import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, Tuple, List

from src.models.rl_agentcontrastivly import RLAgent
from src.models.contrastive_model import ContrastiveLearningAgent
from src.training.contrastive_trainer import ContrastiveTrainer
from src.utils.data_collection import DataCollector


class Contrastiv_RL_agent2(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 1e-3, hidden_dim: int = 256, input_channels: int = 3, latent_dim: int = 128, tau: float = 0.1, epsilon_decay = 0.995, gamma = 0.99):
        super().__init__()
        self.contrastive_model = ContrastiveLearningAgent(input_channels=input_channels, latent_dim=latent_dim, tau=tau)
        self.contrastive_trainer = ContrastiveTrainer(self.contrastive_model, learning_rate=learning_rate)
        self.rl_agent = RLAgent(state_dim, action_dim, hidden_dim=hidden_dim, epsilon_decay = epsilon_decay, gamma = gamma)
        self.data_collector = DataCollector()
        self.optimizer = optim.Adam(self.contrastive_model.parameters(), lr=learning_rate)



    def q_forward(self, x):
        with torch.no_grad():
            x = self.contrastive_model.query_encoder(x.unsqueeze(0))
            return self.rl_agent.q_network(x)


    # functions of contrastive_model
    def compute_infonce_loss(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return ContrasiveEncoder.info_nce_loss(query, key, self.tau)

    def compute_state_entropy(self, state: torch.Tensor, memory_bank: torch.Tensor) -> torch.Tensor:
        return self.contrastive_model.compute_state_entropy(state, memory_bank)

    def query_encoder(self, X):
        return self.contrastive_model.query_encoder(X)

    # functions of contrastive_trainer
    def train_contrastive_model(self, queries: torch.Tensor, keys: torch.Tensor, epochs: int = 100, batch_size: int = 32) -> List[float]:
        return# self.contrastive_trainer.train(queries, keys,epochs = epochs, batch_size = batch_size)

    # functions of rl_agent
    def memory(self):
        return self.rl_agent.memory

    def act(self, state: torch.Tensor) -> int:
        return self.rl_agent.act(state)

    def remember(self, state, action, reward, next_state, done, intrinsic_reward = 0):
        q_state = self.data_collector.default_augmentation(next_state)
        k_state = self.data_collector.default_augmentation(next_state)
        q = self.contrastive_model.query_encoder(q_state.unsqueeze(0))
        with torch.no_grad():
            k = self.contrastive_model.key_encoder(k_state.unsqueeze(0))
        intrinsic_reward = self.contrastive_model.compute_infonce_loss(q,k)
        # Backward pass
        with torch.no_grad():
            self.rl_agent.remember(self.contrastive_model.query_encoder(state.unsqueeze(0)).squeeze(), action, reward, self.contrastive_model.query_encoder(next_state.unsqueeze(0)).squeeze(), done, intrinsic_reward)
        return intrinsic_reward

    def train(self,batch_size: int = 32) -> float:
        return self.rl_agent.train(batch_size)

    def update_target_network(self):
        self.rl_agent.update_target_network()

    def epsilon_decay(self):
        self.rl_agent.decay_epsilon()

    