from src.utils.data_collection import DataCollector
from src.environments.minigrid_wrapper import MiniGridWrapper
import torch
import numpy as np
import random
import os

def set_global_seed(seed: int):
    """Set seeds for Python, NumPy, PyTorch, and CUDA for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
def build_dataset_per_seed(environments, seed: int):
    set_global_seed(seed)
    for difficulty, env_name in environments.items():
        print(f"Collecting data for {difficulty} environment: {env_name}")
        env = MiniGridWrapper(env_name, seed=seed)
        data_collector = DataCollector(env, max_episodes=100)

        obs = data_collector.collect_data()
        queries, keys = data_collector.create_contrastive_pairs(obs, mode="NOISE")
        
        save_dir = f"data/{seed}"
        os.makedirs(save_dir, exist_ok=True)

        
        torch.save({
            'observations': obs,
            'queries': queries,
            'keys': keys
        }, f'data/{seed}/{difficulty}_dataset.pth')
        
        print(f"Saved {len(obs)} observations for {difficulty} task with seed {seed}.")

def main():

    environments = {
        #late choose what grids to actualy use
        'easy': 'MiniGrid-Empty-8x8-v0',
        'moderate': 'MiniGrid-DoorKey-8x8-v0', 
        'hard': 'MiniGrid-MultiRoom-N6-v0'
        
    }
    seeds = []
    with open('src/utils/seeds_pretraining.txt', 'r') as f:
        seeds = [int(line.strip()) for line in f]

    
    for seed in seeds:
        build_dataset_per_seed(environments, seed)

    

if __name__ == "__main__":
    main()
