import gymnasium as gym
import numpy as np
import torch
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, FullyObsWrapper
from typing import Tuple

class MiniGridWrapper():
    
    
    def __init__(self, env_name: str, seed: int, cnn: bool = True):
        """
        Initializes the MiniGrid environment wrapper.
        Args:
            env_name (str): The name of the MiniGrid environment.
            seed (int): The random seed for reproducibility.
        """
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.cnn = cnn
        if cnn:
        
            self.env = RGBImgObsWrapper(self.env)
            self.env = ImgObsWrapper(self.env)
        else:
            
            self.env = FullyObsWrapper(self.env)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.env.reset(seed=seed)
        sample_obs, _ = self.env.reset()
        
        ### this part below is made with claude sonnet 4 ###
        processed_obs = self.preprocess_observation(sample_obs)
        
        # Take a few steps to make sure the observation size is consistent
        for _ in range(3):
            action = self.env.action_space.sample()
            next_obs, _, done, truncated, _ = self.env.step(action)
            next_processed = self.preprocess_observation(next_obs)
            
            if next_processed.shape != processed_obs.shape:
                print(f"Warning: Inconsistent observation shapes detected!")
                print(f"Initial: {processed_obs.shape}, Later: {next_processed.shape}")
            
            if done or truncated:
                sample_obs, _ = self.env.reset()
                processed_obs = self.preprocess_observation(sample_obs)
                break
        
        # Use the processed observation shape for state size calculation
        self._actual_obs_shape = processed_obs.shape
            
        print(f"Observation type: {type(sample_obs)}")
        print(f"Processed observation shape: {self._actual_obs_shape}")
        print(f"Final state size will be: {self.get_state_size()}")
        if isinstance(sample_obs, dict):
            print(f"Observation keys: {sample_obs.keys()}")
            for key, value in sample_obs.items():
                print(f"  {key}: {type(value)} {getattr(value, 'shape', 'no shape')}")
        ### this part above is made with claude sonnet 4 ###
    
    def get_state_size(self) -> int:
        """
        Returns the size of the state space.
        
        Returns:
            int: The size of the state space.
        """
        if self.cnn:
            return self._actual_obs_shape
        else:
            return int(np.prod(self._actual_obs_shape))
    
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
        ### this part below is made with claude sonnet 4 ###
         # Handle dictionary observations
        if isinstance(obs, dict):
            if self.cnn and 'image' in obs:
                # Use image for CNN
                obs_array = obs['image']
            elif not self.cnn:
                # Flatten all dict values for MLP
                flattened_obs = []
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        flattened_obs.extend(value.flatten())
                    elif isinstance(value, (int, float)):
                        flattened_obs.append(value)
                obs_array = np.array(flattened_obs, dtype=np.float32)
            else:
                # Default to using the first array-like value
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        obs_array = value
                        break
                else:
                    raise ValueError("No suitable observation found in dict")
        else:
            obs_array = obs
        ### this part above is made with claude sonnet 4 ###
        # Convert to tensor
        obs_tensor = torch.from_numpy(obs_array).float()
        
        if self.cnn:
            # For CNN: normalize and ensure correct channel order
            obs_tensor = obs_tensor / 255.0
            if obs_tensor.ndim == 3:  # (H, W, C) -> (C, H, W)
                obs_tensor = obs_tensor.permute(2, 0, 1)
        else:
            # For fully observable: ensure it's flattened
            obs_tensor = obs_tensor.flatten()
            
        return obs_tensor
    
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