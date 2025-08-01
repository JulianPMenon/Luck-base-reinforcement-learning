import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ..models.encoder import ContrasiveEncoder
from .contrastive_trainer import ContrastiveTrainer

# This test script was co-generated with the assistance of OpenAI's ChatGPT (GPT-4).
# Prompted by Julian Menon on 30/7/2025, using Claude to design trainer testing for a contrastive learning model.

def test_training_loop():
    """Test the training loop with real data dimensions
    This function initializes a contrastive encoder and trainer, simulates data,
    and runs a training loop to ensure everything works as expected.
    """
    print("Testing training loop...")
    
    # Create data with same dimensions as your real data
    queries = torch.randn(1496, 3, 64, 64)
    keys = torch.randn(1496, 3, 64, 64)
    
    print(f"Data shapes - Queries: {queries.shape}, Keys: {keys.shape}")
    
    # Initialize encoder and trainer
    encoder = ContrasiveEncoder(input_channels=3, hidden_dim=256, latent_dim=128)
    trainer = ContrastiveTrainer(encoder, learning_rate=1e-3)
    
    print("Starting training with 5 epochs for testing...")
    
    # Test with just 5 epochs first
    losses = trainer.train(queries, keys, epochs=5, batch_size=32)
    
    print(f"Training completed. Number of losses returned: {len(losses)}")
    print(f"Losses: {losses}")
    
    if len(losses) == 5:
        print("✅ Training loop working correctly!")
        
        # Now test with more epochs
        print("\nTesting with 20 epochs...")
        losses_long = trainer.train(queries, keys, epochs=20, batch_size=32)
        print(f"Long training completed. Number of losses: {len(losses_long)}")
        print(f"First 5 losses: {losses_long[:5]}")
        print(f"Last 5 losses: {losses_long[-5:]}")
        
    else:
        print(f"❌ Training loop issue - expected 5 losses, got {len(losses)}")

if __name__ == "__main__":
    test_training_loop()