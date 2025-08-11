# test_data_collection.py
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.data_collection import DataCollector
from src.environments.minigrid_wrapper import MiniGridWrapper

def main():
    env = MiniGridWrapper("MiniGrid-Empty-5x5-v0", seed=42)
    collector = DataCollector( max_episodes=10 , max_steps_per_episode=10 )

    data = collector.collect_data(env)
    print(f"Collected {len(data)} observations")

    queries, keys = collector.create_contrastive_pairs(data, mode="NOISE")
    print(f"Contrastive pairs: queries={queries.shape}, keys={keys.shape}")

if __name__ == "__main__":
    main()
