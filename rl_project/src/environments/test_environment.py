import gymnasium as gym
import numpy as np
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

env = gym.make('MiniGrid-Empty-8x8-v0', render_mode="rgb_array")
env = RGBImgObsWrapper(env) # Get pixel observations
env = ImgObsWrapper(env) # Get rid of the 'mission' field
obs, info = env.reset() # This now produces an RGB tensor only

print(f"Observation shape: {obs.shape}")
print(f"Observation type: {type(obs)}")
print(f"Info: {info}\n")

for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    print(f"Step {i+1}: Action={action}, Reward={reward}, Done={done}")
    print(f"    Obs shape: {obs.shape}\n")
    
    if done:
        obs, info = env.reset()
        print(" Environment reset")

try:
    img = env.render()
    print(f"Render output shape: {img.shape if hasattr(img, 'shape') else 'No shape'}\n")
    print("Environment test successful!")
except Exception as e:
    print(f"Render error: {e}")

env.close()