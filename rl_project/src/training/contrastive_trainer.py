import torch
import torch.optim as optim
from typing import List, Tuple
from ..models.contrastive_model import ContrastiveLearningAgent
from ..models.encoder import ContrasiveEncoder

class ContrastiveTrainer:
    
    def __init__(self, model: ContrastiveLearningAgent, learning_rate: float = 1e-3):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def train(self, queries: torch.Tensor, keys: torch.Tensor, epochs: int = 100, batch_size: int = 32) -> List[float]:
        losses = []
        dataset_size = queries.size(0)
        
        print(f"Dataset size: {dataset_size}")
        print(f"Batch size: {batch_size}")
        print(f"Batches per epoch: {dataset_size // batch_size}")
        print(f"Queries device: {queries.device}, Keys device: {keys.device}")
        print(f"Model device: {next(self.model.parameters()).device}")
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # shuffle to ensure that the model does not learn any sequence-specific patterns
            indices = torch.randperm(dataset_size)
            queries_shuffled = queries[indices]
            keys_shuffled = keys[indices]
            
            for i in range(0, dataset_size, batch_size):
                batch_queries = queries_shuffled[i:i+batch_size]
                batch_keys = keys_shuffled[i:i+batch_size]

                # Encode images to feature vectors
                q = self.model.query_encoder(batch_queries)
                k = self.model.key_encoder(batch_keys)

                # Forward pass (InfoNCE loss)
                loss = self.model.compute_infonce_loss(q, k)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update key encoder with EMA
                self.model._update_key_encoder()

                epoch_losses.append(loss.item())
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses
         
                
                