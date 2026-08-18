import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class YawDDclassifier(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        # pretrained resnet model as feature extractor
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1]) # keep only the model backbone and remove the final head
        
        # temporal attention pooling
        self.attn = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # classification head
        self.cls_head = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1), 
        )

    #Note to self: batch only determines how much is worked on in parallel
    def forward(self, x):
        B, T, C, H, W = x.shape #B = Batch Size, T = Time (Nr of Frames), CHW=Color-Height-Width (image)
        
        # frame-wise feature extraction with 2D backbone
        x = x.view(B * T, C, H, W)    # Flatten Time-dimension to process each frame independently
        x = self.feature_extractor(x)           # (B*T, F, 1, 1)
        x = x.view(B, T, -1)                    # (B, T, F)

        # attention pooling over time
        scores = self.attn(x)               # Calculate attention scores Tensor Shape: (B, T, 1)
        weights = torch.softmax(scores, dim=1) # scales scores from 0 to 1, shape unchanged
        pooled = (x * weights).sum(dim=1)   # (B, F), single feature vector per sequence

        # final logits
        logits = self.cls_head(pooled).squeeze(-1)  # (B,)
        return logits