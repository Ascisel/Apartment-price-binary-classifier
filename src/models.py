import torch.nn as nn
import torch
from src.config import ModelConfig as Config


class RealEstateClassifier(nn.Module):

    def __init__(self, num_data_shape: int, cat_data_shape: int = 0):
        super().__init__()
        self.embeding_layers = nn.Sequential(
            nn.Linear(cat_data_shape, cat_data_shape),
            nn.Tanh()
        )
        self.ffnn_layers = nn.Sequential(
            nn.Linear(cat_data_shape + num_data_shape, Config.LINEAR_LAYERS_HIDDEN[0]),
            nn.BatchNorm1d(Config.LINEAR_LAYERS_HIDDEN[0]),
            nn.LeakyReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.LINEAR_LAYERS_HIDDEN[0], Config.LINEAR_LAYERS_HIDDEN[1]),
            nn.BatchNorm1d(Config.LINEAR_LAYERS_HIDDEN[1]),
            nn.LeakyReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.LINEAR_LAYERS_HIDDEN[1], 1),
            nn.Sigmoid()
        )

    def forward(self, num_x: torch.Tensor, cat_x: torch.Tensor) -> torch.Tensor:
        cat_x_embedded = self.embeding_layers(cat_x)
        input = torch.cat([num_x, cat_x_embedded], dim=1)
        output = self.ffnn_layers(input)

        return output