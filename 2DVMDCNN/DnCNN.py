"""
DnCNN
一维网络去噪
"""
import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, in_channels=4, num_classes=1,num_of_layers=20):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv1d(in_channels=in_channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ELU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv1d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm1d(features))
            layers.append(nn.ELU(inplace=True))
        layers.append(nn.Conv1d(in_channels=features, out_channels=num_classes, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        #out = out.unsqueeze(1)
        return {"out": out}
        # return out
