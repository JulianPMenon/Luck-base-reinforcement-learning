import gymnasium as gym
import numpy as np
import torch
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from typing import Tuple

class MiniGridWrapper(gym.Wrapper):
    
    
    def __init__(self, env_name: str, seed: int):
        """
        Initializes the MiniGrid environment wrapper.
        Args:
            env_name (str): The name of the MiniGrid environment.
            seed (int): The random seed for reproducibility.
        """
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.env = RGBImgObsWrapper(self.env)
        self.env = ImgObsWrapper(self.env)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.env.reset(seed=seed)
    
    def reset(self) -> torch.Tensor:
        """
        Resets the environment and returns the initial observation as a tensor.
        
        Returns:
            torch.Tensor: The initial observation of the environment.
        """
        obs, info = self.env.reset()
        return self.preprocess_observation(obs)   
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, bool, dict]:
        """
        Takes a step in the environment.
        
        Returns:
            
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.preprocess_observation(obs)
        done = terminated or truncated
    
        return obs, reward, done, truncated, info
    
    def preprocess_observation(self, obs: np.ndarray) -> torch.Tensor:
        """ 
        Preprocesses the observation from the environment.
        
        """
        obs = torch.from_numpy(obs).float() / 255
        
        if obs.ndim == 3:  # If the observation is an image
            obs = obs.permute(2, 0, 1)  # Change to (C, H, W) format
        elif obs.ndim == 1:  # If the observation is a vector
            obs = obs.unsqueeze(0)  # Add a batch dimension
            
        return obs
    
    def render(self):
        """
        Renders the environment.
        
        Returns:
            np.ndarray: The rendered image of the environment.
        """
        return self.env.render()
    
    def close(self):
        """
        Closes the environment.
        """
        self.env.close()