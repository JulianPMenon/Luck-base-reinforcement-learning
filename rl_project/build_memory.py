import torch
from src.models.contrastive_model import ContrastiveLearningAgent

def build_memory_bank(model, observations, sample_rate=10):
    """Build memory bank for entropy estimation"""
    memory_bank = []
    
    with torch.no_grad():
        for i in range(0, len(observations), sample_rate):
            obs = observations[i]
            if obs.dim() == 3:
                obs = obs.unsqueeze(0)
            
            encoding = model.query_encoder(obs)
            memory_bank.append(encoding.squeeze())
    
    return torch.stack(memory_bank)

# Build memory banks for all difficulties
difficulties = ['easy', 'moderate', 'hard']

for difficulty in difficulties:
    print(f"Building memory bank for {difficulty} task...")
    
    # Load model and data
    model = ContrastiveLearningAgent(input_channels=3, latent_dim=128)
    model.load_state_dict(torch.load(f'models/{difficulty}_contrastive_model.pth'))
    
    data = torch.load(f'data/{difficulty}_dataset.pth')
    observations = data['observations']
    
    # Build memory bank
    memory_bank = build_memory_bank(model, observations, sample_rate=5)
    
    # Save memory bank
    torch.save(memory_bank, f'data/{difficulty}_memory_bank.pth')
    
    print(f"Memory bank saved: {memory_bank.shape[0]} states")