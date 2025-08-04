import torch
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.environments.minigrid_wrapper import MiniGridWrapper
from src.models.rl_agent import RLAgent
from src.utils.metrics import MetricsTracker

# Test with simple environment

seed = 666666 #demo seed for testing
env = MiniGridWrapper('MiniGrid-Empty-5x5-v0', seed=seed, cnn=False)
state_size = np.prod(env.observation_space.shape)
conf = [0.2437, 0.5827, 0.8271, 0.6080, 0.1923, 0.9548]
agent = RLAgent(
    state_size=env.get_state_size(), 
    action_size=env.action_space.n, 
    batch_size=32,
    epsilon = conf[0], 
    epsilon_decay=conf[1], 
    epsilon_min=conf[2], 
    gamma=conf[3], 
    learning_rate=conf[4], 
    intrinsic_weight=conf[5]
)
# Track Q-network weights norm
def get_weights_norm(agent):
    return sum(p.data.norm().item() for p in agent.q_network.parameters())

initial_weights_norm = get_weights_norm(agent)
print(f"Initial Q-network weights norm: {initial_weights_norm:.6f}")
metrics = MetricsTracker()

for episode in range(100):
    observation = env.reset()
    print(f"Episode {episode} - Initial Observation stats: mean={observation.mean():.4f}, std={observation.std():.4f}, shape={observation.shape}")
    state = observation.flatten()  # Normalize the state
    done = False
    steps = 0
    total_reward = 0
    
    while not done and steps < 100:
        action = agent.act(state)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = next_observation.flatten() # Normalize the next state
        
        agent.remember(state, action, reward, next_state, done)
        if len(agent.memory) > agent.batch_size:    
            prev_weights_norm = get_weights_norm(agent)
            loss = agent.train()
            new_weights_norm = get_weights_norm(agent)
            if loss > 0:
                metrics.update_loss(loss_type='rl', loss=loss)
            if steps % 10 == 0:
                print(f"    Q-network weights norm before: {prev_weights_norm:.6f}, after: {new_weights_norm:.6f}, diff: {new_weights_norm - prev_weights_norm:.6f}")
        total_reward += reward
        state = next_state
        steps += 1

        if steps % 10 == 0:
            print(f"  Step {steps} | Reward: {reward:.2f} | Next obs mean: {next_observation.mean():.2f}")
        print(f"Step {steps}: reward={reward}, terminated={terminated}, truncated={truncated}, done={done}")
        
    metrics.update_episode(total_reward, steps, [])
    
    agent.decay_epsilon()
    
    if episode % 10 == 0:
        print(f"[Episode {episode}] Epsilon: {agent.epsilon:.4f}")
    
    if episode % 20 == 0:
        avg_reward = metrics.get_average_return(20)
        print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Total Rewards: {total_reward:.2f}, done: {done}, Steps: {steps}")
    
    print(f"Episode {episode} ended. Total reward: {total_reward}, Steps: {steps}, Done: {done}")

final_weights_norm = get_weights_norm(agent)
print(f"Final Q-network weights norm: {final_weights_norm:.6f}")
print("RL Agent test completed successfully!")