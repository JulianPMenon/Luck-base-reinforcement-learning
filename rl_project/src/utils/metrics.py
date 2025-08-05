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
    
    def update_intrinsic_reward(self, intrinsic_reward: float):
        """Update intrinsic rewards"""
        self.intrinsic_rewards.append(intrinsic_reward)
        
    def get_exploration_efficiency(self) -> float:
        """Calculate exploration efficiency (unique states visited)"""
        return len(self.exploration_counts)
    
    def get_average_return(self, window: int = 100) -> float:
        """Get average return over last window episodes"""
        if len(self.episode_rewards) < window:
            return np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        return np.mean(self.episode_rewards[-window:])
    
    def plot_metrics(self, save_path: str = None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        
        # Losses
        if self.losses['rl']:
            axes[1, 0].plot(self.losses['rl'], label='RL Loss')
        if self.losses['contrastive']:
            axes[1, 0].plot(self.losses['contrastive'], label='Contrastive Loss')
        axes[1, 0].set_title('Training Losses')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        
        # Intrinsic rewards
        if self.intrinsic_rewards:
            axes[1, 1].plot(self.intrinsic_rewards)
            axes[1, 1].set_title('Intrinsic Rewards')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Intrinsic Reward')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
