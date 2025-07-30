import torch
import torch.nn as nn

from rl_project.src.models.predictor import Predictor
from rl_project.src.models.rcnn import RCNN


class RND():
    def __init__(self, seed_rcnn, seed_predictor):
        self.rcnn = RCNN(seed_rcnn)
        self.predictor = Predictor(seed_predictor)
        self.loss_fn = nn.MSELoss()

    def train(self, state):
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.predictor.train()
        state.to(device)
        prediction_x = self.predictor(state)
        prediction_y = self.rcnn(state)
        loss = self.loss_fn(prediction_x,prediction_y)
        loss.backward()
        return loss
    