import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml
from src.environments.minigrid_wrapper import MiniGridWrapper
from src.models.contrastive_model import ContrastiveLearningAgent
from src.models.rl_agent import RLAgent
from src.utils.data_collection import DataCollector
from src.utils.metrics import MetricsTracker
from src.training.contrastive_trainer import ContrastiveTrainer

def load_config(config_path):
    """Load configuration from a YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def run_experiment(config: dict):
    print(f"Running experiment: {config['name']}")
    
    # 1. Initialize environment
    env = MiniGridWrapper(config['env_name'])
    
    # 2. Collect data for contrastive learning
    print("                                           |   mmm  \n"
          "Collecting data for contrastive learning...|  (@_@) \n"
          "                                           | <( : )>\n"
          "                                           |   / \  \n"
          )
    
    data_collector = DataCollector(env, max_episodes=config['data_collection_episodes'])
    observations = data_collector.collect_random_data()
    queries, keys = data_collector.create_contrastive_pairs(observations, mode="Noise")
    queries, keys += data_collector.create_contrastive_pairs(observations, mode="Rotate")
    
    # 3. Initialize contrastive model
    
    
    # 4. Train contrastive model
    
    # 5. Initialize RL agent
    
    # 6. Build memory bank for entropy estimation
    
    # 7. Training
    
    # 8. Evaluation