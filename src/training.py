import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from torchinfo import summary
from tqdm import tqdm  

from src.evaluation import evaluate

from torch.utils.tensorboard import SummaryWriter

import os

import psutil

from src.config import Config

class YawDDclassifier(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        # pretrained resnet model
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1]) # keep only the model backbone and remove the final head
        
        # temporal attention pooling
        self.attn = nn.Sequential(
            nn.Linear(
                backbone.fc.in_features,
                Config.ATTENTION_HIDDEN
            ),
        nn.Tanh(),
        nn.Linear(
            Config.ATTENTION_HIDDEN,
            1
            ),
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
    
def print_ram(tag=""):
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 ** 3)  # Convert bytes to GB
    print(f"{tag} RAM usage: {ram_usage:.2f} GB")


def trainer(trainloader,
            valloader,
            model,
            epochs,
            lr,
            freeze_backbone, 
            device,
            writer=None,
            trial_number=None):

    print_ram("Start trainer")

    os.makedirs("checkpoints", exist_ok=True)
    
    best_f1 = 0
    best_epoch = 0

    # objective function is binary cross entropy loss with logits 
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(Config.POS_WEIGHT))
    
    # set non-trainable parameters
    if freeze_backbone:
        for p in model.feature_extractor.parameters():
            p.requires_grad=False

    # Get trainable parameters and hand to optimizer
    tp = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(tp, lr=lr, weight_decay=Config.WEIGHT_DECAY) # AdamW uses weight decay with default 1e-2
    writer.add_scalar(
        "LearningRate",
        optimizer.param_groups[0]["lr"],
        0
    )
    # summary(model)

    # train loop
    for epoch in range(epochs): 

        print_ram(f"Epoch {epoch} start")

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRAD_CLIP_NORM) # gradient clipping                
            optimizer.step()

            # update running loss
            running_loss += loss.item()

            print_ram(f"Epoch {epoch} after train")

        epoch_loss = running_loss / len(trainloader)
        
        print(f'  Loss: {epoch_loss:0.4f}')

        if writer:
            writer.add_scalar(
                "Loss/Train",
                epoch_loss,
                epoch
            )


        # evaluate train and validation data
        train_metrics = evaluate(trainloader, model, device)

        print_ram(f"Epoch {epoch} after train eval")

        val_metrics = evaluate(valloader, model, device)

        print_ram(f"Epoch {epoch} after val eval")

        if writer:
            for metric in train_metrics:
                writer.add_scalar(
                    f"{metric}/Train",
                    train_metrics[metric],
                    epoch
                )
            for metric in val_metrics:
                writer.add_scalar(
                    f"{metric}/Validation",
                    val_metrics[metric],
                    epoch
                )


        print(f"Train Acc: {train_metrics['accuracy']:.3f}   --   Val Acc: {val_metrics['accuracy']:.3f}")
        print(f"Train F1: {train_metrics['f1']:.3f}   --   Val F1c: {val_metrics['f1']:.3f}")  

        # Save best model checkpoint
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                f"checkpoints/trial_{trial_number}_best.pt"
            )

            if writer:
                writer.add_scalar(
                    "BetsF1",
                    best_f1,
                    epoch
                )

    return best_f1, best_epoch
