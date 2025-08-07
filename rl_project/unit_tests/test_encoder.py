import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.encoder import ContrasiveEncoder
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def test_encoder_shape():
    model = ContrasiveEncoder(input_channels=3)
    x = torch.randn(8, 3, 84, 84)
    y = model(x)
    assert y.shape == (8, 128), f"Got shape {y.shape}"
    
def test_embedding():
    """
        Test the embedding of the ContrasiveEncoder using t-SNE.
        
        
        # This test script was co-generated with the assistance of OpenAI's ChatGPT (GPT-4).
        # Prompted by Julian Menon on 30/7/2025, using ChatGPT to design encoder testing for a contrastive learning model.

    """
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    encoder = ContrasiveEncoder(input_channels=3)
    encoder.eval()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            features = encoder(images).detach().cpu().numpy()
            all_features.append(features)
            all_labels.append(labels.numpy())

            if len(all_features) * images.shape[0] >= 1000:
                break
            
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Step 4: t-SNE projection
    proj = TSNE(n_components=2, perplexity=30, init='pca', random_state=42).fit_transform(features)

    # Step 5: Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    plt.colorbar()
    plt.title("t-SNE of ContrasiveEncoder Features")
    plt.tight_layout()
    plt.savefig("tsne_encoder_output.png")  # Optional: save
    plt.show()
    

   


test_encoder_shape()
test_embedding()
