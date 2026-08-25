import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import f1_score, accuracy_score
from torchinfo import summary
from tqdm import tqdm
import os

from src.evaluation import evaluate


class YawDDclassifier(nn.Module):
    """
    Deep Learning architecture for driver drowsiness detection.
    Combines a spatial ResNet18 backbone with temporal attention pooling.
    """
    def __init__(self, dropout):
        """
        Initializes the model architecture.

        Args:
            dropout (float): Dropout probability for the classification head.
        """
        super().__init__()

        # Pretrained ResNet model
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Keep only the model backbone and remove the final classification head
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        
        # Temporal attention pooling
        self.attn = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Classification head
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
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input video tensor of shape (B, T, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B,).
        """
        B, T, C, H, W = x.shape
        
        # Frame-wise feature extraction with 2D backbone
        x = x.view(B * T, C, H, W)                  # (B*T, C, H, W)
        x = self.feature_extractor(x)               # (B*T, F, 1, 1)
        x = x.view(B, T, -1)                        # (B, T, F)

        # Attention pooling over time
        scores = self.attn(x)                       # (B, T, 1)
        weights = torch.softmax(scores, dim=1)
        pooled = (x * weights).sum(dim=1)           # (B, F)

        # Final logits
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
            tb_writer=None,
            patience=5,
            trial=None
            ):
    """
    Executes the training loop, including validation, checkpointing, and early stopping.

    Args:
        trainloader (DataLoader): DataLoader for training data.
        valloader (DataLoader): DataLoader for validation data.
        model (nn.Module): The model to be trained.
        epochs (int): Maximum number of training epochs.
        lr (float): Learning rate.
        freeze_backbone (bool): If True, freezes the ResNet feature extractor.
        device (torch.device): Device to run the training on (CPU/GPU).
        save_dir (str): Directory path to save model checkpoints.
        trial_params (dict, optional): Hyperparameters used in the current trial.
        tb_writer (SummaryWriter, optional): TensorBoard writer instance.
        patience (int): Number of epochs to wait for validation loss improvement before early stopping.
        trial (optuna.Trial, optional): Current Optuna trial object for pruning.

    Returns:
        tuple: (best_f1, best_epoch) representing the optimal validation performance.
    """
    
    os.makedirs(save_dir, exist_ok=True)

    best_f1 = 0
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0


    # Objective function is binary cross entropy loss with logits 
    pos_weight = torch.tensor(2.0, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    # Set non-trainable parameters
    if freeze_backbone:
        for p in model.feature_extractor.parameters():
            p.requires_grad=False

    # Get trainable parameters and hand to optimizer
    tp = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(tp, lr=lr, weight_decay=1e-2)


    # Train loop
    for epoch in range(epochs): 

        # Initialize running loss and lists for on-the-fly metrics
        running_loss = 0
        all_train_preds = []
        all_train_labels = []
        
        model.train()
        
        # 1. Force BatchNorm layers into eval mode if backbone is frozen
        if freeze_backbone:
            model.feature_extractor.eval()

        for frames, labels in tqdm(trainloader, desc=f'Epoch {epoch}'):
            frames, labels = frames.to(device), labels.to(device)

            # Forward + Backward pass
            optimizer.zero_grad()
            logits = model(frames)          
            loss = criterion(logits, labels)
            loss.backward()        
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)              
            optimizer.step()

            running_loss += loss.item()
            
            # 2. Collect predictions directly within the loop for performance boost
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                all_train_preds.append(preds.cpu())
                all_train_labels.append(labels.cpu())
            
        avg_train_loss = running_loss / len(trainloader)
        
        # Compute training metrics
        y_true_train = torch.cat(all_train_labels).numpy()
        y_pred_train = torch.cat(all_train_preds).numpy()
        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_f1 = f1_score(y_true_train, y_pred_train, zero_division=0)
        
        print(f'  Avg Train Loss: {avg_train_loss:0.4f}')

        # 3. Evaluate validation data and pass criterion to compute validation loss
        val_metrics = evaluate(valloader, model, device, criterion=criterion)

        print(f"Train Acc: {train_acc:.3f}   --   Val Acc: {val_metrics['accuracy']:.3f}")
        print(f"Train F1: {train_f1:.3f}   --   Val F1: {val_metrics['f1']:.3f}")  
        print(f"Val Loss: {val_metrics['loss']:.4f}") 

        # Pass metrics to TensorBoard
        if tb_writer is not None:
            tb_writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            tb_writer.add_scalar("Loss/Val", val_metrics['loss'], epoch)
            tb_writer.add_scalar("Accuracy/Train", train_acc, epoch)
            tb_writer.add_scalar("Accuracy/Val", val_metrics['accuracy'], epoch)
            tb_writer.add_scalar("F1/Train", train_f1, epoch)
            tb_writer.add_scalar("F1/Val", val_metrics['f1'], epoch)

        # Optuna Pruning (based on target metric F1)
        if trial is not None:
            trial.report(val_metrics['f1'], epoch)
            if trial.should_prune():
                print(f"➔ Trial {trial.number} pruned at epoch {epoch} (poor performance).")
                if tb_writer is not None:
                    tb_writer.close()
                raise optuna.TrialPruned()

        # Checkpointing logic (Focus on best F1-Score)
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_metrics['f1'],
                "val_loss": val_metrics['loss']
            }
            
            path = os.path.join(save_dir, "best_model.pth")
            torch.save(checkpoint, path)
            print(f"New best model (F1={best_f1:.3f}) saved at epoch {epoch}!")

        # Early stopping logic (Focus on Overfitting / Validation Loss)
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0 
            print(f"Val Loss improved to {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No validation loss improvement (current: {val_metrics['loss']:.4f}, best: {best_val_loss:.4f}). Patience: {patience_counter}/{patience}")

        # Trigger early stopping
        if patience_counter >= patience:
            print(f"Early Stopping triggered at epoch {epoch} (Val Loss increased). Ending this trial.")
            break
            
    return best_f1, best_epoch

        