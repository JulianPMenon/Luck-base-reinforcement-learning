import torch

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.encoder import ContrasiveEncoder


def test_encoder_shape():
    model = ContrasiveEncoder(input_channels=3)
    x = torch.randn(8, 3, 84, 84)
    y = model(x)
    assert y.shape == (8, 128), f"Got shape {y.shape}"

test_encoder_shape()
