import numpy as np
import torch
from typing import Dict, List
import matplotlib.pyplot as plt

class MetricsTracker:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.episode_rewards = []
        self.episode_lengths = []
        self.exploration_counts = {}
        self.intrinsic_rewards = []
        self.losses = {'rl': [], 'contrastive': []}
        
    def update_episode(self, reward: float, length: int, visited_states: List):
        """Update metrics after an episode"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
    
    def update_loss(self, loss: float, loss_type: str = 'rl'):
        """Update the loss metrics"""
        if loss_type in self.losses:
            self.losses[loss_type].append(loss)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
    def get_average_return(self, window: int = 100) -> float:
        """Get average return over last window episodes"""
        if len(self.episode_rewards) < window:
            return np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        return np.mean(self.episode_rewards[-window:])