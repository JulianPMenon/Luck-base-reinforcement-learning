import torch
import numpy as np
from typing import List, Tuple

class DataCollector:
    
    def __init__(self, env, max_episodes=1000):
        self.env = env
        self.max_episodes = max_episodes
        
    def collect_data(self) -> List[torch.Tensor]:
        """
        Collects data from the environment by running a series of episodes.
        
        Returns:
            List[torch.Tensor]: A list of tensors containing the collected data.
        """
        data = []
        for episode in range(self.max_episodes):
            obs = self.env.reset()
            data.append(torch.tensor(obs, dtype=torch.float32))
            done = False
            
            while not done or not truncated:
                action = self.env.action_space.sample()
                obs, reward, done, truncated, info = self.env.step(action)
                data.append(torch.tensor(obs, dtype=torch.float32))
            
                if len(data) >= self.max_episodes:
                    break
        return data