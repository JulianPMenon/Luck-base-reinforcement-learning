import gymnasium as gym
import numpy as np
import torch
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, FullyObsWrapper, RGBImgPartialObsWrapper
from typing import Tuple

def show_grid(self):
        """
        Displays the current grid using matplotlib and obs['image'].
        Only works if the observation is a dict with an 'image' key (e.g., FullyObsWrapper).
        """
        import matplotlib.pyplot as plt
        obs, _ = self.env.reset()
        if isinstance(obs, dict) and 'image' in obs:
            img = obs['image']
            plt.figure(figsize=(4, 4))
            plt.imshow(img)
            plt.title('MiniGrid Observation (obs["image"])')
            plt.axis('off')
            plt.show()
        else:
            print("Current observation does not have an 'image' key.")
            
env = gym.make('MiniGrid-Empty-8x8-v0')
env = RGBImgPartialObsWrapper(env)
env = ImgObsWrapper(env)

obs = env.reset()
print(f"Observation shape: {obs[0].shape}")
show_grid(env)
env.render()