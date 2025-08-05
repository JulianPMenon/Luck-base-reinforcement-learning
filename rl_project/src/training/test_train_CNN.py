import torch
import torch
import numpy as np
import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..models.encoder import ContrasiveEncoder
from .contrastive_trainer import ContrastiveTrainer
from ..environments.minigrid_wrapper import MiniGridWrapper
from ..utils.data_collection import DataCollector

def set_global_seed(seed: int):
    """Set seeds for Python, NumPy, PyTorch, and CUDA for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def main():
    environments = {
        #late choose what grids to actualy use
        'easy': 'MiniGrid-Empty-8x8-v0',
        'moderate': 'MiniGrid-DoorKey-8x8-v0', 
        'hard': 'MiniGrid-MultiRoom-N6-v0'
    }
    
    print("Initializing environment and data collector...\n")
    env_name = environments['easy']  # Example: using the 'easy' environment
    seed = 42  # Example seed for reproducibility
    set_global_seed(seed)
    
    env = MiniGridWrapper(env_name, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_collector = DataCollector(env, max_episodes=500)
    collected_data = data_collector.collect_data()

    print(f"Collected {len(collected_data)} observations from the environment.\n")
    
    queries, keys = data_collector.create_contrastive_pairs(collected_data)
    print(f"Queries shape: {queries.shape}, dtype: {queries.dtype}")
    print(f"Keys shape: {keys.shape}, dtype: {keys.dtype}")
    queries = queries.to(device)
    keys = keys.to(device)
    
    encoder = ContrasiveEncoder(input_channels=3, hidden_dim=256, latent_dim=128).to(device)
    
    print("Encoder initialized. Starting training...\n")
    
    trainer = ContrastiveTrainer(encoder, learning_rate=1e-3)

    # Reduce epochs for testing, increase batch size if you have enough data
    epochs = 50 if len(collected_data) < 1000 else 100
    batch_size = min(32, len(collected_data) // 10)  # Ensure at least 10 batches

    print(f"Training with {epochs} epochs, batch size {batch_size}")
    losses = trainer.train(queries, keys, epochs=epochs, batch_size=batch_size)

    print(f"Training completed. Total epochs: {len(losses)}")
    print(f"First 5 losses: {losses[:5]}")
    print(f"Last 5 losses: {losses[-5:]}")
    print(f"Final loss: {losses[-1]:.6f}")
    
    torch.save(encoder.state_dict(), "contrastive_encoder_minigrid.pth")
    print("Model saved.\n")
    
if __name__ == "__main__":
    main()