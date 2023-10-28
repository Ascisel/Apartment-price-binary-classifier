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
        # self.emb_layer = nn.Linear(cat_data_shape, cat_data_shape)
        # self.act_emb = nn.Tanh()
        # self.layer1 = nn.Linear(cat_data_shape + num_data_shape, 300)
        # self.bn1 = nn.BatchNorm1d(300)
        # self.act_1 = nn.LeakyReLU()
        # self.d1 = nn.Dropout(0.4)
        # self.layer2 = nn.Linear(300, 150)
        # self.bn2 = nn.BatchNorm1d(150)
        # self.act_2 = nn.LeakyReLU()
        # self.d2 = nn.Dropout(0.4)
        # self.layer3 = nn.Linear(150, 1)
        # self.act_3 = nn.Sigmoid()
    def forward(self, num_x: torch.Tensor, cat_x: torch.Tensor) -> torch.Tensor:
        cat_x_embedded = self.embeding_layers(cat_x)
        input = torch.cat([num_x, cat_x_embedded], dim=1)
        output = self.ffnn_layers(input)

        # cat_x_embedded = self.emb_layer(cat_x)
        # cat_x_embedded = self.act_emb(cat_x_embedded)
        # x = torch.cat([x, cat_x_embedded], dim=1)
        # x = self.layer1(x)
        # x = self.bn1(x)
        # x = self.act_1(x)
        # x = self.d1(x)
        # x = self.layer2(x)
        # x = self.bn2(x)
        # x = self.act_2(x)
        # x = self.d2(x)
        # x = self.layer3(x)
        # x = self.act_3(x)
        
        return output