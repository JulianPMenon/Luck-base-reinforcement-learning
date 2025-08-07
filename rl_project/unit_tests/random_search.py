# Renamed from tests/random_search.py
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.environments.minigrid_wrapper import MiniGridWrapper
from src.models.rl_agent import RLAgent
from src.utils.metrics import MetricsTracker

import numpy as np
import matplotlib.pyplot as plt
import minigrid

# ...existing code...
