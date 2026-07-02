import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from torchinfo import summary
from tqdm import tqdm
import os

from src.evaluation import evaluate


class YawDDclassifier(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        # pretrained resnet model
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
            freeze_backbone, 
            device,
            save_dir="models",
            trial_params=None,
            tb_writer=None):
    
    os.makedirs(save_dir, exist_ok=True)

    best_f1 = 0
    best_epoch = 0


    # objective function is binary cross entropy loss with logits 
    # Change needs future evaluation
    # Old code: criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0))
    pos_weight = torch.tensor(2.0, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    # set non-trainable parameters
    if freeze_backbone:
        for p in model.feature_extractor.parameters():
            p.requires_grad=False

    # Get trainable parameters and hand to optimizer
    tp = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(tp, lr=lr, weight_decay=1e-2) # AdamW uses weight decay with default 1e-2
    
    # summary(model)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping                
            optimizer.step()

            # update running loss
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(trainloader)
        print(f'  Loss: {running_loss:0.4f}')
        print(f'  Avg Loss: {avg_train_loss:0.4f}')

        # evaluate train and validation data
        train_metrics = evaluate(trainloader, model, device)
        val_metrics = evaluate(valloader, model, device)

        print(f"Train Acc: {train_metrics['accuracy']:.3f}   --   Val Acc: {val_metrics['accuracy']:.3f}")
        print(f"Train F1: {train_metrics['f1']:.3f}   --   Val F1: {val_metrics['f1']:.3f}")  

        # Pass metrics to TensorBoard
        if tb_writer is not None:
            tb_writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            tb_writer.add_scalar("Accuracy/Train", train_metrics['accuracy'], epoch)
            tb_writer.add_scalar("Accuracy/Val", val_metrics['accuracy'], epoch)
            tb_writer.add_scalar("F1/Train", train_metrics['f1'], epoch)
            tb_writer.add_scalar("F1/Val", val_metrics['f1'], epoch)

        # Save best model checkpoint
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

            checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_metrics['f1']
                    #"val_acc": val_metrics['accuracy'],
                    #"train_f1": train_metrics['f1'],
                    #"train_acc": train_metrics['accuracy'],
                    #"lr": lr,
                    #"freeze_backbone": freeze_backbone,
                    #"trial_params": trial_params
                }
            
            path = os.path.join(save_dir, "best_model.pth")
            torch.save(checkpoint, path)
            print(f"✅ Saved new best model (F1={best_f1:.3f}) at epoch {epoch}")
            
    return best_f1, best_epoch

        