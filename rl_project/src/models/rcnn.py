import torch
import torch.nn as nn
import torch.nn.functional as F

class RCNN(nn.Module):
    """
    A random convolutional network 
    """
    def __init__(self, seed: int, pool_size: int = 4, input_channels: int = 3, hidden_dim: int = 256, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        torch.manual_seed(seed)        
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

    def forward_featues(self, x: torch.Tensor) -> torch.Tensor:
        # since this is a random Model it shouldnt be trained
        with torch.no_grad():
            self.eval()
            features = self.conv_layers(x)
            return features.view(features.size(0), -1) # Flatten the features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.eval()
            z = self.forward_featues(x)
            return F.normalize(z, dim=1)  # Normalize the output