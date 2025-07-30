import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrasiveEncoder(nn.Module):
    """
    A neural network encoder for contrastive learning.
    
    This encoder processes input observations and outputs a feature vector.
    """
    
    def __init__(self, pool_size: int = 4, input_channels: int = 3, hidden_dim: int = 256, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_dim = 128 * pool_size * pool_size
        
        # CNN layers for visual feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((pool_size, pool_size))
        )
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2), #prevent overfitting a little
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward_featues(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(x)
        return features.view(features.size(0), -1) # Flatten the features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_featues(x)
        z = self.projection_head(x)
        return F.normalize(z, dim=1)  # Normalize the output for InfoNCE
    
    def info_nce_loss(query, key, temperature: float = 0.1):
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        logits = torch.mm(query, key.t()) / temperature
        labels = torch.arange(query.size(0), device=query.device)
        return F.cross_entropy(logits, labels)