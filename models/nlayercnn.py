import torch
import torch.nn as nn


class NLayerCNN(nn.Module):

    def __init__(self, in_channels, out_channels, num_filters, num_layers=3):
        super().__init__()

        sequence = [
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        ]
        for l in range(1, num_layers):
            mult_prev = min(2 ** (l - 1), 8)
            mult = min(2 ** l, 8)
            sequence += [
                nn.Conv2d(num_filters * mult_prev, num_filters * mult, 3, padding=1),
                nn.BatchNorm2d(num_filters * mult),
                nn.ReLU(inplace=True)
            ]
        sequence += [
            nn.Conv2d(num_filters * mult, out_channels, 1),
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
