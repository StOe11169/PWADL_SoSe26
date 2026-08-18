import torch
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


def evaluate(
    loader,
    model,
    device,
    threshold,
    writer=None,
    epoch=None,
    prefix="val",
    verbose=True
):
    """
    Evaluiert ein Modell auf einem DataLoader.

    Args:
        loader:
            PyTorch DataLoader.

        model:
            Zu evaluierendes Modell.

        device:
            CPU oder GPU.

        threshold:
            Schwellenwert für binäre Klassifikation.
            Beispiel: 0.3

        writer:
            Optionaler TensorBoard SummaryWriter.

        epoch:
            Aktuelle Epoche für TensorBoard.

        prefix:
            Prefix für TensorBoard-Tags, z.B.:
            "train", "val", "test"

        verbose:
            Wenn True, werden Metriken in der Konsole ausgegeben.

    Returns:
        Dictionary mit:
            accuracy
            precision
            recall
            f1
            roc_auc
            pr_auc
            confusion_matrix
    """

    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for frames, labels in tqdm(loader, desc=f"Evaluate {prefix}", leave=False):
            frames = frames.to(device)
            labels = labels.to(device).float().view(-1)

            logits = model(frames).view(-1)
            probs = torch.sigmoid(logits)

            preds = (probs > threshold).float()

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    y_true = torch.cat(all_labels).numpy().astype(int)
    y_pred = torch.cat(all_preds).numpy().astype(int)
    y_prob = torch.cat(all_probs).numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    # ROC-AUC und PR-AUC können nur berechnet werden,
    # wenn im Ground Truth beide Klassen vorkommen.
    if len(set(y_true.tolist())) == 2:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
    }

    if verbose:
        print(f"\n[{prefix}] Ergebnisse")
        print(f"Accuracy : {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall   : {recall:.3f}")
        print(f"F1       : {f1:.3f}")

        if roc_auc == roc_auc:
            print(f"ROC-AUC  : {roc_auc:.3f}")
        else:
            print("ROC-AUC  : nan")

        if pr_auc == pr_auc:
            print(f"PR-AUC   : {pr_auc:.3f}")
        else:
            print("PR-AUC   : nan")

        print("Confusion Matrix:")
        print(cm)

        print("Vorhergesagte Klassen:", sorted(set(y_pred.tolist())))
        print("Wahre Klassen        :", sorted(set(y_true.tolist())))

    if writer is not None and epoch is not None:
        writer.add_scalar(f"{prefix}/Accuracy", accuracy, epoch)
        writer.add_scalar(f"{prefix}/Precision", precision, epoch)
        writer.add_scalar(f"{prefix}/Recall", recall, epoch)
        writer.add_scalar(f"{prefix}/F1", f1, epoch)

        if roc_auc == roc_auc:
            writer.add_scalar(f"{prefix}/ROC_AUC", roc_auc, epoch)

        if pr_auc == pr_auc:
            writer.add_scalar(f"{prefix}/PR_AUC", pr_auc, epoch)

    return metrics