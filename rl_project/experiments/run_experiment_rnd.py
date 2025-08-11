import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import yaml
import copy
from src.environments.minigrid_wrapper import MiniGridWrapper
from src.utils.data_collection import DataCollector
from src.utils.metrics import MetricsTracker
from src.models.rnd_ import RND
import torch.optim as optim
import torch.nn.functional as F
from unit_tests import hyperband

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_experiment_rnd(config, metrics, lr , agent, max_episodes=0):
    
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Running RND baseline: {config['name']}")
    env = MiniGridWrapper(config['env_name'], seed=42)
    rnd = agent.to(device)
    optimizer = optim.Adam(rnd.predictor_network.parameters(), lr=lr)
    max_episode = config['rl_episodes'] if max_episodes == 0 else max_episodes
    rnd_losses = []
    best_avg_reward = float('-inf')
    best_weights = None
    best_weights_path = None
    
    for episode in range(max_episode+1):
        obs = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < config['max_steps_per_episode']:
            # Convert numpy to torch if needed
            if isinstance(obs, (np.ndarray,)):
                obs = torch.from_numpy(obs)
            obs_tensor = obs.unsqueeze(0).permute(0, 3, 1, 2).to(device)
            
            # Compute intrinsic reward
            intrinsic_reward = rnd.compute_intrinsic_reward(obs_tensor).item()
            
            # Take random action
            action_key = np.random.choice(list(config['actionmap'].keys()))
            next_obs, reward, terminated, truncated, _ = env.step(config['actionmap'][action_key])
            done = terminated or truncated
            
            # Convert numpy to torch if needed
            if isinstance(next_obs, (np.ndarray,)):
                next_obs = torch.from_numpy(next_obs)
            next_obs_tensor = next_obs.unsqueeze(0).permute(0, 3, 1, 2).to(device)
            target_features, predicted_features = rnd(next_obs_tensor)
            loss = F.mse_loss(predicted_features, target_features.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            rnd_losses.append(loss.item())
            metrics.update_loss(loss_type='rnd', loss=loss.item())
            total_reward += reward + intrinsic_reward
            steps += 1
            obs = next_obs
            metrics.update_intrinsic_reward(intrinsic_reward=intrinsic_reward)
            
        metrics.update_episode(total_reward, steps, [])
        
        if episode % 10 == 0 and episode != 0:
            avg_reward = metrics.get_average_return()
            avg_rnd_loss = np.mean(rnd_losses[-steps:]) if steps > 0 else 0.0
            print(f"[RND] Episode {episode}: Total Reward: {total_reward}, Avg Reward: {avg_reward:.2f}, Avg RND Loss: {avg_rnd_loss:.4f}")
            # Track and save best average reward and weights
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_weights = copy.deepcopy(rnd.state_dict())
                best_weights_path = f"results/{config['name']}/rnd/best_weights.pth"
                torch.save(best_weights, best_weights_path)
    
    result_dir = f"results/{config['name']}/rnd"
    os.makedirs(result_dir, exist_ok=True)
    metrics.plot_metrics(save_path=f"{result_dir}/metrics.png")
    torch.save(rnd.predictor_network.state_dict(), f"{result_dir}/rnd_predictor.pth")
    print(f"RND baseline completed. Results saved to {result_dir}")
    
    return {
        'agent': rnd,
        'config': config,
        'metrics': metrics,
        'avg_reward': metrics.get_average_return(),
        'best_avg_reward': best_avg_reward,
        'best_weights_path': best_weights_path,
    }

if __name__ == "__main__":
    config = load_config('rl_project/experiments/configs/moderate_task_lava_gap.yaml')
    result_dir = f"results/{config['name']}/rnd"
    with open(result_dir+"/results.txt", 'a') as f:
        for seed in range(10):
            print(seed)
            torch.manual_seed(seed)
            feature_dim = 128  
            rnd = RND(input_channels=3, feature_dim=feature_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            metrics = MetricsTracker()
            population = run_experiment_rnd(config,agent=rnd, metrics=metrics, lr=0.00015, max_episodes=400)
            #hyperband.plot_heatMap(pop = {'agent': population['agent']}, env = MiniGridWrapper(population['config']['env_name'], seed= 42), actionmap=population['config']['actionmap'],save_path=f"{result_dir}/heat_map_{seed}_.png")
            population['metrics'].plot_metrics(save_path=f"{result_dir}/metrics_{seed}_.png")
            print(f"seed: {seed} | avg_reward: {population['avg_reward']} | best_avg_reward: {population['best_avg_reward']} | best_weights: {population['best_weights_path']}")
            f.write(f"seed: {seed} | avg_reward: {population['avg_reward']} | best_avg_reward: {population['best_avg_reward']} | best_weights: {population['best_weights_path']}\n")
            torch.save(rnd.state_dict(), f"{result_dir}/rl_agent_{seed}.pth")


