# pipeline/fusion_model.py
# Input: 384 (text) + 98 (video) + 42 (audio) = 524

import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, input_dim=524):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x) * 100  # Scale to 0-100 for scoring