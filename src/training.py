import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from torchinfo import summary
from tqdm import tqdm  

from src.evaluation import evaluate


def trainer(trainloader, valloader, model, epochs, lr, freeze_backbone, device):
    
    best_f1 = 0
    best_epoch = 0

    # objective function is binary cross entropy loss with logits 
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0)) #missclafiying yawns is twice as costly 
    
    # set non-trainable parameters
    if freeze_backbone:
        for p in model.feature_extractor.parameters():
            p.requires_grad=False

    # Get trainable parameters and hand to optimizer
    tp = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(tp, lr=lr, weight_decay=1e-2) # AdamW uses weight decay with default 1e-2, currently hardcoded, change that
    
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
        
        print(f'  Loss: {running_loss:0.4f}')

        # evaluate train and validation data
        train_metrics = evaluate(trainloader, model, device)
        val_metrics = evaluate(valloader, model, device)

        print(f"Train Acc: {train_metrics['accuracy']:.3f}   --   Val Acc: {val_metrics['accuracy']:.3f}")
        print(f"Train F1: {train_metrics['f1']:.3f}   --   Val F1c: {val_metrics['f1']:.3f}")  

        # Save best model checkpoint
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch

    return best_f1, best_epoch

        