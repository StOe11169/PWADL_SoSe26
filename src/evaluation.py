import torch
import pandas as pd
from tqdm import tqdm  
from sklearn.metrics import accuracy_score , precision_score, recall_score, f1_score


def evaluate(loader, model, device, criterion=None, input_key="frames"):

    
    #Set model to evaluation mode. I.e no dropout, dont compute gradients etc.
    model.eval()
    with torch.no_grad():
        all_labels = []
        all_preds = []
        total_loss = 0
        total_samples = 0

        for batch in tqdm(loader):
            #shift data to device
            inputs = batch[input_key].to(device)
            labels = batch["labels"].to(device)

            # forward pass
            logits = model(inputs)
            probs = torch.sigmoid(logits) #probability threshold of 0.5 equals a logit threshold of zero
            preds = (probs > 0.5).float()

            #accumulate batch-mean losses; NOT a sample-weighted epoch mean
            if criterion is not None:
                loss = criterion(logits, labels)
                batch_samples = labels.numel()
                total_loss += loss.item() * batch_samples
                total_samples += batch_samples
            
            # save labels and predictions
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

        if not all_labels:
            raise RuntimeError("Cannot evaluate an empty DataLoader.")
        
        # convert to numpy
        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()   
            
        # return metrics
        results = {
            'accuracy':  accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall':    recall_score(y_true, y_pred, zero_division=0),
            'f1':        f1_score(y_true, y_pred, zero_division=0),
        }
        #add loss if available
        if criterion is not None:
            results["loss"] = total_loss / total_samples

        #raw data for confusion matrix
        results["y_true"] = y_true
        results["y_pred"] = y_pred

        return results


@torch.inference_mode() #https://docs.pytorch.org/docs/2.9/notes/autograd.html#inference-mode -> avoids some overhead during fusion preds
def predict_logits(loader, model, device, input_key):
    #return raw logits for late fusion, one row for each video
    model.eval()
    rows = []

    for batch in tqdm(loader, desc=f"Predicting {input_key}", leave=False):
        inputs = batch[input_key].to(device)

        #save logits before classification
        logits = model(inputs).cpu()

        labels = batch["labels"].cpu()
        filepath = batch["filepath"]

        #save one prediction per video, keep filepath for alignment
        for filepath, label, logit in zip(filepath, labels, logits):
            rows.append({"filepath": filepath, "label": int(label.item()), "logit": float(logit.item())}) #Note:label and logit are pytorch tensors

    return pd.DataFrame(rows)