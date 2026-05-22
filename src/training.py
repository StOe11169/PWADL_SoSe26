import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm  

from src.evaluation import evaluate


class YawDDclassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # pretrained resnet model
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1]) # keep only the model backbone and remove the final head
        
        # temporal attention pooling
        self.attn = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
        )

        # classification head
        self.cls_head = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        
        # frame-wise feature extraction with 2D backbone
        x = x.view(B * T, C, H, W)    # (B*T, C, H, W)
        x = self.feature_extractor(x)           # (B*T, F, 1, 1)
        x = x.view(B, T, -1)                    # (B, T, F)

        # attention pooling over time
        scores = self.attn(x)               # (B, T, 1)
        weights = torch.softmax(scores, dim=1)
        pooled = (x * weights).sum(dim=1)   # (B, F)

        # final logits
        logits = self.cls_head(pooled).squeeze(-1)  # (B,)
        return logits
    

def trainer(trainloader,
            valloader,
            model,
            epochs,
            lr,
            device):

    # optimizer 
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train loop
    for epoch in range(epochs):

        # init running loss
        running_loss = 0

        # go through all data
        model.train()
        for frames, labels in tqdm(trainloader, desc=f'Epoch {epoch}'):
            frames, labels = frames.to(device), labels.to(device) # shift data to device

            # forward + backward pass
            optimizer.zero_grad()
            logits = model(frames)           
            loss    = criterion(logits, labels)
            loss.backward()                   
            optimizer.step()

            # update running loss
            running_loss += loss.item() * frames.size(0)
        
        print(f'  Loss: {running_loss:0.4f}')

        # evaluate train and validation data
        train_metrics = evaluate(trainloader, model, device)
        val_metrics = evaluate(valloader, model, device)

        print(f"Train Acc: {train_metrics['accuracy']:.3f}   --   Val Acc: {val_metrics['accuracy']:.3f}")  

        
