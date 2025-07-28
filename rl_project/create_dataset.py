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

def main():

    SEED = 42
    set_global_seed(SEED)
    
    environments = {
        #late choose what grids to actualy use
        'easy': 'MiniGrid-Empty-8x8-v0',
        'moderate': 'MiniGrid-DoorKey-8x8-v0', 
        'hard': 'MiniGrid-MultiRoom-N6-v0'
        
    }

    for difficulty, env_name in environments.items():
        print(f"Collecting data for {difficulty} environment: {env_name}")
        env = MiniGridWrapper(env_name, seed=SEED)
        data_collector = DataCollector(env, max_episodes=100)

        obs = data_collector.collect_data()
        queries, keys = data_collector.create_contrastive_pairs(obs, mode="NOISE")
        
        torch.save({
            'observations': obs,
            'queries': queries,
            'keys': keys
        }, f'data/{difficulty}_dataset.pth')
        
        print(f"Saved {len(obs)} observations for {difficulty} task")

if __name__ == "__main__":
    main()
