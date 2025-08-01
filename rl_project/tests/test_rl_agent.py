import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.environments.minigrid_wrapper import MiniGridWrapper
from src.models.rl_agent import RLAgent
from src.utils.metrics import MetricsTracker

# Test with simple environment

seed = 666666 #demo seed for testing
env = MiniGridWrapper('MiniGrid-Empty-8x8-v0', seed=seed)
state_size = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
agent = RLAgent(state_size=state_size, action_size=env.action_space.n, batch_size=32, epsilon_decay= 0.97)
metrics = MetricsTracker()

for episode in range(100):
    observation = env.reset()
    print(f"Episode {episode} - Initial Observation stats: mean={observation.mean():.4f}, std={observation.std():.4f}, shape={observation.shape}")
    state = observation.flatten() / 255.0  # Normalize the state
    done = False
    steps = 0
    total_reward = 0
    
    while not done and steps < 100:
        action = agent.act(state)
        next_observation, reward, done, truncated, _ = env.step(action)
        next_state = next_observation.flatten() / 255.0  # Normalize the next state
        
        agent.remember(state, action, reward, next_state, done)
        if len(agent.memory) > agent.batch_size:    
            loss = agent.train()
            if loss > 0:
                metrics.update_loss(loss_type='rl', loss=loss)
                
        total_reward += reward
        state = next_state
        steps += 1

        if steps % 10 == 0:
            print(f"  Step {steps} | Reward: {reward:.2f} | Next obs mean: {next_observation.mean():.2f}")
        
    metrics.update_episode(total_reward, steps, [])

    if episode % 20 == 0:
        avg_reward = metrics.get_average_return(20)
        print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Total Rewards: {total_reward:.2f}, done: {done}, Steps: {steps}")

        
        
        
        


print("RL Agent test completed successfully!")