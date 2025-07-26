import gym
import gym_minigrid
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

env = gym.make('MiniGrid-Empty-8x8-v0')
env = RGBImgPartialObsWrapper(env)
env = ImgObsWrapper(env)

obs = env.reset()
print(f"Observation shape: {obs.shape}")
env.render()