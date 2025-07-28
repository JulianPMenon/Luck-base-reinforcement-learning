import torch
import numpy as np
from typing import List, Tuple
import random
import torchvision.transforms.functional as TF

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
            #data.append(torch.tensor(obs, dtype=torch.float32))
            data.append(obs.detach().clone())
            done = False
            
            while not done:
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                data.append(obs.detach().clone())
                #data.append(torch.tensor(obs, dtype=torch.float32))
                
                done = terminated or truncated
                if len(data) >= self.max_episodes:
                    break
        return data
    
    def create_contrastive_pairs(self, data: List[torch.Tensor], aug = None, mode: str = "NOISE") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates contrastive pairs from the collected data.
        
        Args:
            data (List[torch.Tensor]): The collected data.
            aug: Optional augmentation function to apply to the data.
            mode (str): The augmentation mode, can be "NOISE" or "ROTATE". 
                It is used in the default_augmentation method.
        
        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: A list of tuples containing contrastive pairs.
        """
        if aug is None:
            aug = lambda obs: self.default_augmentation(obs, mode=mode)

            
        queries = []
        keys = []
        
        for obs in data:
            
            query = aug(obs).to(obs.device)
            key = aug(obs).to(obs.device)
                
            queries.append(query)
            keys.append(key)
            
        return torch.stack(queries), torch.stack(keys)
    
    def default_augmentation(self, obs: torch.Tensor, mode: str = "NOISE") -> torch.Tensor:
        """
        Default augmentation function that applies random noise to the observation.
        
        Args:
            obs (torch.Tensor): The observation tensor.
            mode (str): The augmentation mode, can be "NOISE" or "ROTATE".
                "NOISE" applies random noise, "ROTATE" applies a random rotation.
                "ROTATE" rotates the tensores to a random 90 degree increment angle.
        
        Returns:
            torch.Tensor: The augmented observation tensor.
        """
        if mode == "NOISE":
            noise = torch.randn_like(obs) * 0.1
            augmented = obs + noise
            
            augmented = torch.clamp(augmented, 0, 1)
            return augmented
        elif mode == "ROTATE":
            angle = random.choice([0, 90, 180, 270])
            augmented = TF.rotate(obs, angle)
            return augmented
        else:
            raise ValueError(f"Unsupported augmentation mode: {mode}")
        
        