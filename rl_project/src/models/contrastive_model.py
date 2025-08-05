import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from .encoder import ContrasiveEncoder

class ContrastiveLearningAgent(nn.Module):

    def __init__(self, input_channels: int = 3, latent_dim: int = 128, tau: float = 0.1):
        super().__init__()
        self.tau = tau  # temperature parameter
        self.latent_dim = latent_dim
        
        self.query_encoder = ContrasiveEncoder(input_channels, latent_dim=latent_dim)
        self.key_encoder = ContrasiveEncoder(input_channels, latent_dim=latent_dim)
        
        self._init_key_encoder()
        
        self.momentum = 0.999
        
    def _init_key_encoder(self):
        """Initialize key encoder with query encoder parameters"""
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
    def _update_key_encoder(self):
        """Update key encoder using EMA"""
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
            
    def forward(self, query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for contrastive learning"""
        q = self.query_encoder(query)
        k = self.key_encoder(key)
        return q, k
    
    def compute_infonce_loss(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return ContrasiveEncoder.info_nce_loss(query, key, self.tau)
    
    def compute_state_entropy(self, state: torch.Tensor, memory_bank: torch.Tensor) -> torch.Tensor:
        """Compute state entropy for intrinsic reward calculation"""
        with torch.no_grad():
            state_encoding = self.query_encoder(state)
            
            # Compute similarities with memory bank
            similarities = torch.mm(state_encoding, memory_bank.t())
            
            # Estimate density using k-nearest neighbors
            k = min(10, memory_bank.size(0))
            if k > 0:
                top_k_similarities, _ = torch.topk(similarities, k, dim=1)
                density = torch.mean(top_k_similarities, dim=1)
                
                # Convert density to entropy (higher entropy for lower density)
                entropy = -torch.log(density + 1e-8)
                return entropy
            else:
                return torch.ones(state_encoding.size(0), device=state.device)
        
        