#https://www.codegenes.net/blog/yamnet-pytorch/
# because am lazy

import torch
import torch.nn as nn
 
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
 
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
 
 
class YamNetLikeModel(nn.Module):
    def __init__(self, num_classes=521):
        super(YamNetLikeModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.separable_conv1 = DepthwiseSeparableConv(32, 64)
        # Add more layers as per the actual YamNet architecture
        self.fc = nn.Linear(1024, num_classes)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.separable_conv1(x)
        # Forward through other layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
