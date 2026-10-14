import torch
from torch import nn


class LetterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        incoming = 1
        for outgoing in [32, 64, 128]:
            layers.extend([nn.Conv2d(incoming, outgoing, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)])
            incoming = outgoing
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Linear(128, 32)

    def forward(self, x):
        return self.output(self.pool(self.features(x)).flatten(1))
