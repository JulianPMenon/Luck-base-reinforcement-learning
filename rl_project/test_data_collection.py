# test_data_collection.py

from src.utils.data_collection import DataCollector
from src.environments.minigrid_wrapper import MiniGridWrapper

def main():
    env = MiniGridWrapper("MiniGrid-Empty-5x5-v0", seed=42)
    collector = DataCollector(env, max_episodes=10)

    data = collector.collect_data()
    print(f"Collected {len(data)} observations")

    queries, keys = collector.create_contrastive_pairs(data, mode="NOISE")
    print(f"Contrastive pairs: queries={queries.shape}, keys={keys.shape}")

if __name__ == "__main__":
    main()
