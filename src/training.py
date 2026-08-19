# Betriebssystemfunktionen für Pfade und Ordnerverwaltung
import os
# PyTorch-Grundmodul für Tensoroperationen und Modellverwaltung
import torch
# Neuronale Netzwerkmodule von PyTorch
import torch.nn as nn
# Optimierungsalgorithmen und Lernraten-Scheduler von PyTorch
import torch.optim as optim
# ResNet18-Architektur und vortrainierte Gewichte aus torchvision
from torchvision.models import resnet18, ResNet18_Weights
# Fortschrittsbalken für Trainingsschleifen
from tqdm import tqdm
# TensorBoard-Writer für Logging von Loss, Metriken und Modellgraph
from torch.utils.tensorboard import SummaryWriter
# Eigene Evaluierungsfunktion für Trainings-, Validierungs- und Testmetriken
from src.evaluation import evaluate
# Hilfsfunktion zum effizienten Auslesen der Labels aus Dataset-Objekten
from src.data import get_labels_from_dataset


class YawDDclassifier(nn.Module):
    """
    Modell für die binäre Gähn-Erkennung in Videosequenzen.

    Eingabe:
        Tensor mit Form (Batch, Time, Channels, Height, Width)

    Architektur:
        - ResNet18 zur Feature-Extraktion pro Frame
        - temporaler Attention-Mechanismus zur Frame-Gewichtung
        - Fully-Connected-Klassifikationskopf für binäre Ausgabe
    """

    def __init__(self, dropout):
        """
        Initialisiert Backbone, Attention-Modul und Klassifikationskopf.

        Args:
            dropout (float): Dropout-Rate zur Regularisierung.
        """
        # Initialisierung der nn.Module-Basisklasse
        super().__init__()
        # Vortrainiertes ResNet18 mit ImageNet-Gewichten laden
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Finale Fully-Connected-Schicht entfernen und nur Feature-Extraktor behalten
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        # Feature-Dimension des ResNet18-Ausgangs bestimmen
        feature_dim = backbone.fc.in_features  # bei ResNet18: 512

        # Temporales Attention-Modul zur Berechnung eines Gewichts pro Frame definieren
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Klassifikationskopf für die binäre Entscheidung definieren
        self.cls_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
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
        Führt den Vorwärtsdurchlauf des Modells aus.

        Args:
            x (torch.Tensor): Eingabetensor mit Form (B, T, C, H, W).

        Returns:
            torch.Tensor: Logits mit Form (B,).
        """
        # Dimensionen des Eingabetensors auslesen
        B, T, C, H, W = x.shape
        # Zeitdimension in die Batchdimension falten, damit ResNet einzelne Frames verarbeitet
        x = x.view(B * T, C, H, W)
        # ResNet-Features für jeden Frame extrahieren
        x = self.feature_extractor(x)
        # Features wieder als Sequenz pro Video darstellen
        x = x.view(B, T, -1)
        # Attention-Scores für jedes Frame-Feature berechnen
        scores = self.attn(x)
        # Attention-Scores über die Zeitachse zu Gewichten normalisieren
        weights = torch.softmax(scores, dim=1)
        # Frame-Features anhand der Attention-Gewichte zu einem Clip-Feature aggregieren
        pooled = (x * weights).sum(dim=1)
        # Aggregiertes Clip-Feature durch den Klassifikationskopf führen
        logits = self.cls_head(pooled).squeeze(-1)

        # Logits für die binäre Klassifikation zurückgeben
        return logits


def compute_pos_weight_from_dataset(dataset, device):
    """
    Berechnet die Positiv-Gewichtung für BCEWithLogitsLoss aus dem aktuellen Trainingssplit.

    Die Gewichtung berücksichtigt ein mögliches Klassenungleichgewicht.

    Args:
        dataset: Trainingsdataset oder Dataset-Wrapper.
        device: Zielgerät für den resultierenden Tensor.

    Returns:
        torch.Tensor: Positiv-Gewichtung für die Loss-Funktion.
    """
    # Labels aus dem aktuellen Dataset extrahieren, ohne Videos zu laden
    labels = get_labels_from_dataset(dataset)
    # Anzahl positiver Beispiele bestimmen
    num_pos = sum(1 for y in labels if float(y) == 1.0)
    # Anzahl negativer Beispiele bestimmen
    num_neg = sum(1 for y in labels if float(y) == 0.0)

    # Sonderfall behandeln, falls keine positiven Beispiele im Split vorhanden sind
    if num_pos == 0:
        print("WARNUNG: Keine positiven Beispiele im Trainingssplit. pos_weight=1.0 gesetzt.")
        pos_weight = 1.0

    # Positiv-Gewichtung als Verhältnis negativer zu positiver Beispiele berechnen
    else:
        pos_weight = num_neg / num_pos

    # Klassenverteilung und berechnete Gewichtung zur Kontrolle ausgeben
    print(
        f"Klassen im Trainingssplit: "
        f"positive={num_pos}, negative={num_neg}, pos_weight={pos_weight:.3f}"
    )

    # Positiv-Gewichtung als Tensor auf dem Zielgerät zurückgeben
    return torch.tensor(pos_weight, dtype=torch.float32, device=device)


def apply_freeze_backbone(model, freeze_backbone):
    """
    Steuert, ob der ResNet-Backbone trainiert oder teilweise eingefroren wird.

    Args:
        model: Zu trainierendes Modell.
        freeze_backbone (int): 0 für vollständiges Training, 1 für teilweises Einfrieren.
    """

    # Prüfen, ob der Backbone eingefroren werden soll
    if freeze_backbone:

        # Alle Parameter des Feature-Extractors zunächst einfrieren
        for p in model.feature_extractor.parameters():
            p.requires_grad = False

        # Letzten ResNet-Block layer4 wieder freigeben
        for p in model.feature_extractor[-2].parameters():
            p.requires_grad = True

        # Statusmeldung für die Konsole ausgeben
        print("Backbone eingefroren, layer4 bleibt trainierbar.")

    # Falls der Backbone nicht eingefroren wird, alle Parameter trainierbar setzen
    else:

        # Alle Parameter des Feature-Extractors für das Training freigeben
        for p in model.feature_extractor.parameters():
            p.requires_grad = True

        # Statusmeldung für die Konsole ausgeben
        print("Kompletter Backbone trainierbar.")


def trainer(
    trainloader,
    valloader,
    model,
    epochs,
    lr,
    freeze_backbone,
    device,
    threshold,
    patience=10,
    log_dir="runs",
    train_eval_loader=None,
    save_path="best_model.pt",
    early_stopping=True
):
    """
    Trainiert das Modell mit optionaler Validierung, Early Stopping und TensorBoard-Logging.

    Args:
        trainloader: DataLoader mit augmentierten Trainingsdaten.
        valloader: DataLoader für Validierungsdaten oder None beim finalen Training.
        model: Zu trainierendes Modell.
        epochs (int): Maximale Anzahl an Trainingsepochen.
        lr (float): Lernrate.
        freeze_backbone (int): Steuert das Einfrieren des Backbones.
        device: Zielgerät für Training und Evaluation.
        threshold (float): Schwellenwert für binäre Vorhersagen.
        patience (int): Anzahl erlaubter Epochen ohne Verbesserung.
        log_dir (str): TensorBoard-Logverzeichnis.
        train_eval_loader: Optionaler Loader für Trainingsmetriken ohne Augmentation.
        save_path (str): Speicherpfad für Modellgewichte.
        early_stopping (bool): Aktiviert Early Stopping anhand des Validation-F1.

    Returns:
        tuple: Bester F1-Score und zugehörige Epoche.
    """
    # Zielordner für Checkpoints aus dem Speicherpfad extrahieren
    save_dir = os.path.dirname(save_path)

    # Checkpoint-Ordner erstellen, falls ein Ordner angegeben ist
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # TensorBoard-Writer für diesen Trainingslauf initialisieren
    writer = SummaryWriter(log_dir=log_dir)

    # Modell auf das Zielgerät verschieben
    model = model.to(device)

    # Backbone je nach Hyperparameter trainierbar setzen oder teilweise einfrieren
    apply_freeze_backbone(model, freeze_backbone)

    # Positiv-Gewichtung aus dem aktuellen Trainingssplit berechnen
    pos_weight = compute_pos_weight_from_dataset(trainloader.dataset, device)

    # Binäre Loss-Funktion mit Logits und Positiv-Gewichtung definieren
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Nur trainierbare Parameter an den Optimizer übergeben
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # AdamW-Optimizer mit Weight Decay initialisieren
    optimizer = optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=1e-2
    )

    # Scheduler zunächst deaktiviert initialisieren
    scheduler = None

    # Scheduler nur verwenden, wenn echte Validierungsdaten vorhanden sind
    if valloader is not None:

        # Lernrate reduzieren, wenn sich der Validation-F1 nicht verbessert
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.7,
            patience=5
        )

    # Modellgraph optional in TensorBoard speichern
    try:

        # Modell für das Graph-Logging in den Evaluationsmodus setzen
        model.eval()

        # Dummy-Forward ohne Gradientenberechnung durchführen
        with torch.no_grad():

            # Einen Batch aus dem Trainingsloader entnehmen
            frames, _ = next(iter(trainloader))

            # Eingabebatch auf das Zielgerät verschieben
            dummy_input = frames.to(device)

            # Modellgraph in TensorBoard speichern
            writer.add_graph(model, dummy_input)

        # Modell wieder in den Trainingsmodus setzen
        model.train()

    # Graph-Logging bei Fehlern überspringen, da es für das Training nicht erforderlich ist
    except Exception as e:
        print(f"TensorBoard add_graph übersprungen: {e}")

    # Besten F1-Score initialisieren
    best_f1 = -1.0

    # Epoche des besten Modells initialisieren
    best_epoch = 0

    # Zähler für Epochen ohne Verbesserung initialisieren
    epochs_no_improve = 0

    # Über alle Trainingsepochen iterieren
    for epoch in range(epochs):

        # Modell in den Trainingsmodus setzen
        model.train()

        # Laufenden Trainingsloss für die Epoche initialisieren
        running_loss = 0.0

        # Über alle Trainingsbatches iterieren
        for frames, labels in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}"):

            # Videoframes auf das Zielgerät verschieben
            frames = frames.to(device)

            # Labels auf das Zielgerät verschieben und in passende Form bringen
            labels = labels.to(device).float().view(-1)

            # Alte Gradienten aus dem vorherigen Optimierungsschritt löschen
            optimizer.zero_grad()

            # Vorwärtsdurchlauf ausführen und Logits berechnen
            logits = model(frames).view(-1)

            # Loss zwischen Logits und Ground-Truth-Labels berechnen
            loss = criterion(logits, labels)

            # Rückwärtsdurchlauf zur Gradientenberechnung ausführen
            loss.backward()

            # Gradientennorm begrenzen, um instabiles Training zu vermeiden
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Modellparameter anhand der berechneten Gradienten aktualisieren
            optimizer.step()

            # Batch-Loss zum laufenden Epochenloss addieren
            running_loss += loss.item()

        # Durchschnittlichen Trainingsloss der Epoche berechnen
        avg_loss = running_loss / len(trainloader)

        # Epochenfortschritt ausgeben
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Durchschnittlichen Trainingsloss ausgeben
        print(f"Train Loss: {avg_loss:.4f}")

        # Trainingsloss in TensorBoard loggen
        writer.add_scalar("Loss/train", avg_loss, epoch)

        # Platzhalter für Trainingsmetriken initialisieren
        train_metrics = None

        # Trainingsmetriken optional ohne Augmentation berechnen
        if train_eval_loader is not None:

            # Modell auf den nicht augmentierten Trainingsdaten evaluieren
            train_metrics = evaluate(
                train_eval_loader,
                model,
                device,
                threshold,
                writer=writer,
                epoch=epoch,
                prefix="train",
                verbose=False
            )

            # Trainings-F1 zur Kontrolle ausgeben
            print(f"\nTrain F1: {train_metrics['f1']:.3f}")

        # Platzhalter für Validierungsmetriken initialisieren
        val_metrics = None

        # Validierung nur durchführen, wenn ein Validierungsloader vorhanden ist
        if valloader is not None:

            # Modell auf Validierungsdaten evaluieren
            val_metrics = evaluate(
                valloader,
                model,
                device,
                threshold,
                writer=writer,
                epoch=epoch,
                prefix="val",
                verbose=False
            )

            # Validierungs-Accuracy ausgeben
            print(f"Val Accuracy: {val_metrics['accuracy']:.3f}")

            # Validierungs-F1 ausgeben
            print(f"Val F1      : {val_metrics['f1']:.3f}")

            # Validierungs-Recall ausgeben
            print(f"Val Recall  : {val_metrics['recall']:.3f}")

            # Validierungs-Precision ausgeben
            print(f"Val Precision: {val_metrics['precision']:.3f}")

            # Scheduler anhand des Validation-F1 aktualisieren
            if scheduler is not None:
                scheduler.step(val_metrics["f1"])

        # Aktuelle Lernrate aus allen Optimizer-Parametergruppen loggen
        for param_group in optimizer.param_groups:

            # Aktuelle Lernrate auslesen
            current_lr = param_group["lr"]

            # Lernrate in TensorBoard speichern
            writer.add_scalar("LearningRate", current_lr, epoch)

            # Lernrate in der Konsole ausgeben
            print(f"Current LR: {current_lr}")

        # Early Stopping nur verwenden, wenn echte Validierungsdaten vorhanden sind
        if early_stopping and valloader is not None:

            # Aktuellen Validation-F1 als Zielmetrik auswählen
            current_f1 = val_metrics["f1"]

            # Prüfen, ob sich der Validation-F1 verbessert hat
            if current_f1 > best_f1:

                # Besten F1-Score aktualisieren
                best_f1 = current_f1

                # Beste Epoche speichern
                best_epoch = epoch

                # Zähler für fehlende Verbesserung zurücksetzen
                epochs_no_improve = 0

                # Aktuelle Modellgewichte als bestes Modell speichern
                torch.save(model.state_dict(), save_path)

                # Speicherhinweis ausgeben
                print(f"Neues bestes Modell gespeichert: {save_path}")

            # Falls keine Verbesserung vorliegt, Zähler erhöhen
            else:
                epochs_no_improve += 1
                print(f"Keine Verbesserung seit {epochs_no_improve} Epoche(n).")

            # Training abbrechen, wenn die Patience überschritten wurde
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping nach {epoch + 1} Epochen.")
                break

        # Alternative Speicherung für finales Training ohne Validierungsdaten
        else:

            # Modell nach jeder Epoche speichern
            torch.save(model.state_dict(), save_path)

            # Aktuelle Epoche als letzte gespeicherte Epoche setzen
            best_epoch = epoch

            # Trainings-F1 als Rückgabewert verwenden, falls er berechnet wurde
            if train_metrics is not None:
                best_f1 = train_metrics["f1"]

    # Bestes Modell nach Early Stopping wieder laden
    if early_stopping and valloader is not None and os.path.exists(save_path):

        # Gespeicherte Modellgewichte laden
        model.load_state_dict(torch.load(save_path, map_location=device))

        # Hinweis zum geladenen besten Modell ausgeben
        print(f"Bestes Modell aus {save_path} zurückgeladen.")

    # TensorBoard-Writer schließen
    writer.close()

    # Besten F1-Score und zugehörige Epoche zurückgeben
    return best_f1, best_epoch
