import torch
from tqdm import tqdm  
from sklearn.metrics import accuracy_score , precision_score, recall_score, f1_score


def evaluate(loader,
            model,
            device,
            criterion=None): # NEU: criterion Parameter
    
    model.eval()
    running_loss = 0.0 # NEU: Variable für den Validation Loss

    with torch.no_grad():
        all_labels = []
        all_preds = []

        for frames, labels in tqdm(loader):
            frames, labels = frames.to(device), labels.to(device) # shift data to device

            # forward pass
            logits = model(frames)
            
            # Validation Loss berechnen, falls criterion übergeben wurde
            if criterion is not None:
                loss = criterion(logits, labels)
                running_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # save labels and predictions
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            
        # convert to numpy
        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()   
            
        metrics = {
            'accuracy':  accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall':    recall_score(y_true, y_pred, zero_division=0),
            'f1':        f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Den durchschnittlichen Validation Loss an das Dictionary anhängen
        if criterion is not None:
            metrics['loss'] = running_loss / len(loader)
            
        return metrics