import torch
import torch.nn as nn
import torch.nn.functional as F

class RND(nn.Module):
    def __init__(self, input_channels = 3, feature_dim = 512):
        
        self.target_network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, feature_dim),
        )
        
        self.predictor_network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, feature_dim),
        )
        
        for i in self.target_network.parameters():
            i.requires_grad = False
            
    def forward(self, x):
        self.eval
        target_features = self.target_network(x)
        predicted_features = self.predictor_network(x)
        
       
        
        return target_features, predicted_features
    
    def compute_intrinsic_reward(self, x):
        self.eval()
        target_features, predicted_features = self.forward(x)
        intrinsic_reward = F.mse_loss(predicted_features, target_features, reduction='none')
        return intrinsic_reward.mean(dim=1)  # Average over batch and spatial dimensions