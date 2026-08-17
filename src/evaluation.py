import torch
from tqdm import tqdm # Fortschrittsbalken für bessere Nutzerinteraktion
from sklearn.metrics import accuracy_score , precision_score, recall_score, f1_score # Metriken-Berechnung
from torch.utils.tensorboard import SummaryWriter # Für das Logging von Metriken in TensorBoard

def evaluate(loader,
            model,
            device,
            threshold,
            writer = None, # Optional: TensorBoard-Logger für Metriken-Logging
            epoch = None): # Optional: Aktuelle Epoche für TensorBoard
    """
    Evaluiert ein trainiertes Modell auf einem gegebenen Datensatz und berechnet wichtige Metriken.
    Kann optional die Ergebnisse in TensorBoard loggen.

    Args:
        loader (DataLoader): Dataloader mit den zu evaluierenden Daten
        model (YawDDclassifier): Das trainierte Modell für die Vorhersagen
        device (torch.device): Gerät (CPU oder GPU) auf dem das Modell läuft
        threshold (float): Entscheidungsgrenze für die binäre Klassifikation (0.25-0.35)
        writer (SummaryWriter, optional): TensorBoard-Logger für das Speichern von Metriken
        epoch (int, optional): Aktuelle Epoche für TensorBoard-Logging

    Returns:
        dict: Dictionary mit den berechneten Metriken:
              {'accuracy': float, 'precision': float, 'recall': float, 'f1': float}
    """
    
    # Modell in Evaluationsmodus setzen - wichtig für Dropout und BatchNorm
    model.eval()
    # Keine Gradienten berechnen
    with torch.no_grad():
        all_labels = [] # Liste zum Sammeln aller Ground-Truth-Labels
        all_preds = [] # Liste zum Sammeln aller Vorhersagen

        # Fortschrittsbalken für die Evaluation anzeigen
        for frames, labels in tqdm(loader):
            # Daten auf das richtige Gerät (CPU/GPU) verschieben
            frames, labels = frames.to(device), labels.to(device)

            # Vorwärtsdurchlauf: Modell berechnet Logits für die Eingabesequenzen
            # logits hat Form (Batch-Größe, 1) - Rohwerte ohne Aktivierung
            logits = model(frames)

            # Sigmoid-Aktivierung: Wandelt Logits in Wahrscheinlichkeiten zwischen 0 und 1 um
            # probs gibt die Wahrscheinlichkeit an, dass die Sequenz Gähnen enthält
            probs = torch.sigmoid(logits)

            # Binäre Vorhersagen: Vergleich der Wahrscheinlichkeiten mit dem Schwellenwert
            # preds ist ein Tensor mit Werten 0 (Kein Gähnen) oder 1 (Gähnen)
            preds = (probs > threshold).float()
           
            
            # Labels und Vorhersagen zur späteren Berechnung der Metriken speichern
            # .cpu() verschiebt die Daten auf die CPU für die Berechnung mit NumPy
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            
        # Alle Batches zu einem großen Tensor kombinieren
        # y_true: Alle Ground-Truth-Labels als NumPy-Array
        # y_pred: Alle Vorhersagen als NumPy-Array
        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy() 


        # Ausgabe der einzigartigen Vorhersagen zur Debugging-Hilfe
        # Zeigt, wie viele verschiedene Klassen das Modell tatsächlich vorhersagt (Sollte [0;1] ergeben, sonst keine Unterscheidung)
        all_preds_tensor = torch.cat(all_preds)
        print("Vorhergesagte Klassen:", all_preds_tensor.unique())
        
        
        
        # Berechnung der Metriken mit scikit-learn
        # zero_division=0 verhindert Fehler, falls eine Klasse nicht vorkommt
        metrics = {
            'accuracy':  accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall':    recall_score(y_true, y_pred, zero_division=0),
            'f1':        f1_score(y_true, y_pred, zero_division=0),
        }

        # Optional: Metriken in TensorBoard loggen, falls Writer und Epoche übergeben wurden
        if writer is not None and epoch is not None:
            writer.add_scalar('Metrics/Accuracy', metrics['accuracy'], epoch)
            writer.add_scalar('Metrics/Precision', metrics['precision'], epoch)
            writer.add_scalar('Metrics/Recall', metrics['recall'], epoch)
            writer.add_scalar('Metrics/F1', metrics['f1'], epoch)

        # Dictionary mit allen berechneten Metriken zurückgeben
        return metrics