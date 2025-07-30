import torch
import torch.optim as optim
from typing import List, Tuple
from ..models.encoder import ContrasiveEncoder

class ContrastiveTrainer:
    
    def __init__(self, encoder: ContrasiveEncoder, learning_rate: float = 1e-3):
        self.model = encoder
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def train(self, queries: torch.Tensor, keys: torch.Tensor, epochs: int = 100, batch_size: int = 32) -> List[float]:
        losses = []
        dataset_size = queries.size(0)
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # shuffle to ensure that the model does not learn any sequence-specific patterns
            indices = torch.randperm(dataset_size)
            queries_shuffled = queries[indices]
            keys_shuffled = keys[indices]
            
            for i in range(0, dataset_size, batch_size):
                batch_queries = queries_shuffled[i:i+batch_size]
                batch_keys = keys_shuffled[i:i+batch_size]
                
                # forward
                loss = self.model.info_nce_loss(batch_queries, batch_keys)
                
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
                
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)    
        
        if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
                
        return losses            
                
                