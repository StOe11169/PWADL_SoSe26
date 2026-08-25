import torch
from tqdm import tqdm  
from sklearn.metrics import accuracy_score , precision_score, recall_score, f1_score


def evaluate(loader,
            model,
            device,
            criterion=None):
    """
    Evaluates the model on a given dataset loader.

    Args:
        loader (DataLoader): The PyTorch DataLoader for validation/test data.
        model (nn.Module): The neural network model to evaluate.
        device (torch.device): The device (CPU/GPU) to run evaluation on.
        criterion (nn.Module, optional): The loss function. If provided, validation loss is computed.

    Returns:
        dict: A dictionary containing evaluation metrics ('accuracy', 'precision', 'recall', 'f1', and optionally 'loss').
    """
    
    model.eval()
    running_loss = 0.0 # Initialize running validation loss

    with torch.no_grad():
        all_labels = []
        all_preds = []

        for frames, labels in tqdm(loader):
            frames, labels = frames.to(device), labels.to(device) # Shift data to device

            # Forward pass
            logits = model(frames)
            
            # Compute validation loss if a criterion is provided
            if criterion is not None:
                loss = criterion(logits, labels)
                running_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.35).float() 
            
            # Save labels and predictions
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            
        # Convert to numpy arrays
        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()   
            
        metrics = {
            'accuracy':  accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall':    recall_score(y_true, y_pred, zero_division=0),
            'f1':        f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Append average validation loss to the dictionary if applicable
        if criterion is not None:
            metrics['loss'] = running_loss / len(loader)
            
        return metrics