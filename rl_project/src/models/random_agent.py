import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.environments.minigrid_wrapper import MiniGridWrapper

# Test with random agent in MiniGrid

def run_random_agent(episodes=20, max_steps=100, seed=42):
    env = MiniGridWrapper('MiniGrid-Empty-5x5-v0', seed=seed, cnn=False)
    total_rewards = []
    for episode in range(episodes):
        obs = env.reset()
        state = obs.flatten()
        done = False
        steps = 0
        total_reward = 0
        while not done and steps < max_steps:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_obs.flatten()
            steps += 1
            print(f"Episode {episode} Step {steps}: reward={reward}, terminated={terminated}, truncated={truncated}, done={done}")
        total_rewards.append(total_reward)
        print(f"Episode {episode} ended. Total reward: {total_reward}, Steps: {steps}, Done: {done}")
    print(f"Average reward over {episodes} episodes: {np.mean(total_rewards):.2f}")

if __name__ == "__main__":
    run_random_agent()
