import torch
import sys
import os
import numpy as np
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.metrics import MetricsTracker
from experiments.run_experiment_rnd import run_experiment_rnd
import json

def create_rnd_config(rl_episodes):
    # No random seed for stochastic search
    # Log-uniform for learning rate
    lr = 10 ** np.random.uniform(-5, -3.3)  # ~1e-5 to ~5e-4
    feature_dim = int(np.random.choice([64, 128, 192]))

    return {
        'name': "easy_task_rnd",
        'env_name': "MiniGrid-Empty-8x8-v0",
        'rl_episodes': rl_episodes,
        'max_steps_per_episode': 300,
        'feature_dim': feature_dim,
        'batch_size': 32,
        'learning_rate': lr,
        'actionmap': {0: 0, 1: 1, 2: 2, 3: 3, 4: 5},
    }

def create_rnd_agent(config):
    metrics = MetricsTracker()
    return {'config': config, 'metrics': metrics, 'avg_reward': 0.0}

def getreward(d):
    return d["avg_reward"]


if __name__ == '__main__':
    torch.manual_seed(0)  
    exponet = 5
    budget = 2 ** (exponet - 1)
    configs = [create_rnd_config(budget) for _ in range(budget)]
    population = [create_rnd_agent(config=config) for config in configs]
    all_results = []

    for n in range(budget):
        for position, pop in enumerate(population):
            print(f"Hyperband RND | Budget: {2**n * 50} | {position+1} of {len(population)}")
            pop['metrics'] = MetricsTracker()
            result = run_experiment_rnd(pop['config'], pop['metrics'], max_episodes=2**n * 50)
            pop['avg_reward'] = result.get_average_return(20)
            all_results.append({'config': pop['config'], 'avg_reward': pop['avg_reward']})
        population.sort(key=getreward, reverse=True)
        # Save and print best config for this budget round
        best_config = population[0]['config']
        best_reward = population[0]['avg_reward']
        print(f"[Checkpoint] Budget round {n}: Best config: {best_config}, Best avg_reward: {best_reward}")
        checkpoint_dir = f"results/{best_config['name']}_hyperband/budget_{n}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(f"{checkpoint_dir}/best_config.json", "w") as f:
            json.dump({'config': best_config, 'avg_reward': best_reward}, f, indent=2)
        
        if n < budget - 1 and len(population) > 1:
            population = population[:int(len(population) / 2)]
    # Save all results
    
    result_dir = f"results/{population[0]['config']['name']}_hyperband"
    os.makedirs(result_dir, exist_ok=True)
    with open(f"{result_dir}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    population[0]['metrics'].plot_metrics(save_path=f"{result_dir}/metrics.png")
    print("Best config:", population[0]['config'])
