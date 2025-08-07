import torch
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.encoder import ContrasiveEncoder
from src.environments.minigrid_wrapper import MiniGridWrapper

# Load model and memory bank
model = ContrasiveEncoder(input_channels=3, latent_dim=128)
model.load_state_dict(torch.load('models/easy_contrastive_model.pth'))
memory_bank = torch.load('data/easy_memory_bank.pth')

# Test environment
env = MiniGridWrapper('MiniGrid-Empty-8x8-v0')

# Collect some test states and compute intrinsic rewards
intrinsic_rewards = []
for _ in range(50):
    obs = env.reset()
    
    # Compute intrinsic reward
    entropy = model.compute_state_entropy(obs.unsqueeze(0), memory_bank)
    intrinsic_rewards.append(entropy.item())
    
    print(f"State entropy (intrinsic reward): {entropy.item():.4f}")

print(f"Average intrinsic reward: {sum(intrinsic_rewards)/len(intrinsic_rewards):.4f}")
print(f"Intrinsic reward std: {torch.tensor(intrinsic_rewards).std().item():.4f}")