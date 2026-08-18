# PyTorch wird für Tensoroperationen, Inferenz und Geräteverwaltung verwendet
import torch
# tqdm zeigt während der Evaluation einen Fortschrittsbalken an
from tqdm import tqdm
# scikit-learn stellt die verwendeten Klassifikationsmetriken bereit
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
    Evaluiert ein Modell auf einem gegebenen DataLoader.

    Args:
        loader: DataLoader mit den zu evaluierenden Daten.
        model: Zu evaluierendes PyTorch-Modell.
        device: Zielgerät für die Berechnung, z. B. CPU oder GPU.
        threshold: Schwellenwert für die binäre Klassifikation.
        writer: Optionaler TensorBoard SummaryWriter.
        epoch: Optionale Epoche für TensorBoard-Logging.
        prefix: Prefix für TensorBoard-Tags, z. B. train, val oder test.
        verbose: Steuert die Ausgabe der Metriken in der Konsole.

    Returns:
        dict: Dictionary mit Accuracy, Precision, Recall, F1, ROC-AUC,
              PR-AUC und Confusion Matrix.
    """

    # Modell in den Evaluationsmodus setzen, damit Dropout und BatchNorm korrekt arbeiten
    model.eval()

    # Liste zum Sammeln aller Ground-Truth-Labels initialisieren
    all_labels = []
    # Liste zum Sammeln aller binären Vorhersagen initialisieren
    all_preds = []
    # Liste zum Sammeln aller vorhergesagten Wahrscheinlichkeiten initialisieren
    all_probs = []

    # Gradientenberechnung deaktivieren, da während der Evaluation nicht trainiert wird
    with torch.no_grad():
        # Über alle Batches des DataLoaders iterieren
        for frames, labels in tqdm(loader, desc=f"Evaluate {prefix}", leave=False):
            # Videoframes auf das Zielgerät verschieben
            frames = frames.to(device)
            # Labels auf das Zielgerät verschieben und in eindimensionale Float-Tensoren umformen
            labels = labels.to(device).float().view(-1)

            # Modellvorhersage als Logits berechnen
            logits = model(frames).view(-1)
            # Logits mit Sigmoid in Wahrscheinlichkeiten der positiven Klasse umwandeln
            probs = torch.sigmoid(logits)

            # Wahrscheinlichkeiten anhand des Schwellwerts in binäre Vorhersagen umwandeln
            preds = (probs > threshold).float()

            # Labels auf die CPU verschieben und für spätere Metrikberechnung speichern
            all_labels.append(labels.cpu())
            # Binäre Vorhersagen auf die CPU verschieben und speichern
            all_preds.append(preds.cpu())
            # Wahrscheinlichkeiten auf die CPU verschieben und speichern
            all_probs.append(probs.cpu())

    # Alle Label-Batches zusammenführen und als Integer-Array für scikit-learn bereitstellen
    y_true = torch.cat(all_labels).numpy().astype(int)
    # Alle Vorhersage-Batches zusammenführen und als Integer-Array bereitstellen
    y_pred = torch.cat(all_preds).numpy().astype(int)
    # Alle Wahrscheinlichkeiten zusammenführen und als NumPy-Array bereitstellen
    y_prob = torch.cat(all_probs).numpy()

    # Accuracy als Anteil korrekt klassifizierter Beispiele berechnen
    accuracy = accuracy_score(y_true, y_pred)
    # Precision berechnen und Division-durch-null-Fälle robust behandeln
    precision = precision_score(y_true, y_pred, zero_division=0)
    # Recall berechnen und Division-durch-null-Fälle robust behandeln
    recall = recall_score(y_true, y_pred, zero_division=0)
    # F1-Score aus Precision und Recall berechnen
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion Matrix in der Form [[TN, FP], [FN, TP]] berechnen
    cm = confusion_matrix(y_true, y_pred)

    # ROC-AUC und PR-AUC nur berechnen, wenn beide Klassen im Ground Truth vorkommen
    if len(set(y_true.tolist())) == 2:
        # ROC-AUC auf Basis der Wahrscheinlichkeiten berechnen
        roc_auc = roc_auc_score(y_true, y_prob)
        # Average Precision als Fläche unter der Precision-Recall-Kurve berechnen
        pr_auc = average_precision_score(y_true, y_prob)
    # Falls nur eine Klasse vorkommt, sind ROC-AUC und PR-AUC nicht sinnvoll definiert
    else:
        # ROC-AUC als NaN markieren
        roc_auc = float("nan")
        # PR-AUC als NaN markieren
        pr_auc = float("nan")

    # Alle berechneten Metriken in einem Dictionary zusammenfassen
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
    }

    # Metriken optional in der Konsole ausgeben
    if verbose:
        # Überschrift mit dem aktuellen Evaluationsprefix ausgeben
        print(f"\n[{prefix}] Ergebnisse")
        # Accuracy formatiert ausgeben
        print(f"Accuracy : {accuracy:.3f}")
        # Precision formatiert ausgeben
        print(f"Precision: {precision:.3f}")
        # Recall formatiert ausgeben
        print(f"Recall   : {recall:.3f}")
        # F1-Score formatiert ausgeben
        print(f"F1       : {f1:.3f}")

        # ROC-AUC nur numerisch ausgeben, wenn der Wert nicht NaN ist
        if roc_auc == roc_auc:
            print(f"ROC-AUC  : {roc_auc:.3f}")
        else:
            print("ROC-AUC  : nan")

        # PR-AUC nur numerisch ausgeben, wenn der Wert nicht NaN ist
        if pr_auc == pr_auc:
            print(f"PR-AUC   : {pr_auc:.3f}")
        else:
            print("PR-AUC   : nan")

        # Confusion Matrix ausgeben
        print("Confusion Matrix:")
        print(cm)

        # Tatsächlich vorhergesagte Klassen zur Kontrolle ausgeben
        print("Vorhergesagte Klassen:", sorted(set(y_pred.tolist())))
        # Im Ground Truth enthaltene Klassen zur Kontrolle ausgeben
        print("Wahre Klassen        :", sorted(set(y_true.tolist())))

    # Metriken optional in TensorBoard schreiben, wenn Writer und Epoche vorhanden sind
    if writer is not None and epoch is not None:
        writer.add_scalar(f"{prefix}/Accuracy", accuracy, epoch)
        writer.add_scalar(f"{prefix}/Precision", precision, epoch)
        writer.add_scalar(f"{prefix}/Recall", recall, epoch)
        writer.add_scalar(f"{prefix}/F1", f1, epoch)

        # ROC-AUC nur loggen, wenn der Wert definiert ist
        if roc_auc == roc_auc:
            writer.add_scalar(f"{prefix}/ROC_AUC", roc_auc, epoch)

        # PR-AUC nur loggen, wenn der Wert definiert ist
        if pr_auc == pr_auc:
            writer.add_scalar(f"{prefix}/PR_AUC", pr_auc, epoch)

    # Dictionary mit allen Metriken zurückgeben
    return metrics