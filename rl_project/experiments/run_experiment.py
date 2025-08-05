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
    # 1. Initialize environment
    env = MiniGridWrapper(config['env_name'])
    
    # 2. Collect data for contrastive learning
    
    # 3. Initialize contrastive model
    
    # 4. Train contrastive model
    
    # 5. Initialize RL agent
    
    # 6. Build memory bank for entropy estimation
    
    # 7. Training
    
    # 8. Evaluation