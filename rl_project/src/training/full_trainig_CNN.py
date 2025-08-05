import torch
from src.models.contrastive_model import ContrastiveLearningAgent
from src.training.contrastive_trainer import ContrastiveTrainer
import matplotlib.pyplot as plt

difficulties = ['easy', 'moderate', 'hard']
seed = 42  # For reproducibility
for difficulty in difficulties:
    print(f"Training contrastive model for {difficulty} task...")
    
    # Load dataset
    data = torch.load(f'data/{seed}/{difficulty}_dataset.pth')
    queries, keys = data['queries'], data['keys']
    
    # Initialize model
    model = ContrastiveLearningAgent(input_channels=3, latent_dim=128)
    trainer = ContrastiveTrainer(model, lr=0.001)
    
    # Train model
    losses = trainer.train(queries, keys, epochs=100, batch_size=32)
    
    # Save model
    torch.save(model.state_dict(), f'models/{difficulty}_contrastive_model.pth')
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(f'Contrastive Learning Loss - {difficulty.capitalize()} Task')
    plt.xlabel('Epoch')
    plt.ylabel('InfoNCE Loss')
    plt.savefig(f'results/{difficulty}_contrastive_loss.png')
    plt.close()
    
    print(f"Model saved for {difficulty} task. Final loss: {losses[-1]:.4f}")