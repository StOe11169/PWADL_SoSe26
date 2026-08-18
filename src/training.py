import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.evaluation import evaluate
from src.data import get_labels_from_dataset


class YawDDclassifier(nn.Module):
    """
    Modell für binäre Gähn-Erkennung:

    Eingabe:
        Video-Clip mit Form:
        (Batch, Time, Channels, Height, Width)

    Architektur:
        - ResNet18 pro Frame
        - Temporale Attention über Frames
        - Klassifikationskopf für binäre Ausgabe
    """

    def __init__(self, dropout):
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Entfernt nur die finale Fully-Connected-Schicht.
        # Enthalten bleiben:
        # conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        feature_dim = backbone.fc.in_features  # bei ResNet18: 512

        self.attn = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

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
        Args:
            x:
                Tensor mit Form (B, T, C, H, W)

        Returns:
            logits:
                Tensor mit Form (B,)
        """

        B, T, C, H, W = x.shape

        # Frames einzeln durch ResNet schicken:
        # (B, T, C, H, W) -> (B*T, C, H, W)
        x = x.view(B * T, C, H, W)

        # ResNet-Features:
        # (B*T, C, H, W) -> (B*T, 512, 1, 1)
        x = self.feature_extractor(x)

        # Zurück zur Sequenz:
        # (B*T, 512, 1, 1) -> (B, T, 512)
        x = x.view(B, T, -1)

        # Attention-Scores pro Frame:
        # (B, T, 512) -> (B, T, 1)
        scores = self.attn(x)

        # Gewichte über Zeitachse normalisieren
        weights = torch.softmax(scores, dim=1)

        # Gewichtete Summe über Frames:
        # (B, T, 512) -> (B, 512)
        pooled = (x * weights).sum(dim=1)

        # Binäre Logits:
        # (B, 512) -> (B,)
        logits = self.cls_head(pooled).squeeze(-1)

        return logits


def compute_pos_weight_from_dataset(dataset, device):
    """
    Berechnet pos_weight für BCEWithLogitsLoss dynamisch aus dem aktuellen Trainingssplit.

    pos_weight = Anzahl negative Beispiele / Anzahl positive Beispiele

    Vorteil:
        Funktioniert korrekt für:
        - train
        - trainval
        - Cross-Validation-Folds
        - eigene zusätzliche Videos
    """

    labels = get_labels_from_dataset(dataset)

    num_pos = sum(1 for y in labels if float(y) == 1.0)
    num_neg = sum(1 for y in labels if float(y) == 0.0)

    if num_pos == 0:
        print("WARNUNG: Keine positiven Beispiele im Trainingssplit. pos_weight=1.0 gesetzt.")
        pos_weight = 1.0
    else:
        pos_weight = num_neg / num_pos

    print(
        f"Klassen im Trainingssplit: "
        f"positive={num_pos}, negative={num_neg}, pos_weight={pos_weight:.3f}"
    )

    return torch.tensor(pos_weight, dtype=torch.float32, device=device)


def apply_freeze_backbone(model, freeze_backbone):
    """
    freeze_backbone = 0:
        gesamter ResNet-Backbone wird trainiert.

    freeze_backbone = 1:
        Backbone wird weitgehend eingefroren,
        aber layer4 bleibt trainierbar.

    Wichtig:
        feature_extractor[-1] ist avgpool und hat keine trainierbaren Parameter.
        feature_extractor[-2] ist layer4.
    """

    if freeze_backbone:
        for p in model.feature_extractor.parameters():
            p.requires_grad = False

        # layer4 wieder freigeben
        for p in model.feature_extractor[-2].parameters():
            p.requires_grad = True

        print("Backbone eingefroren, layer4 bleibt trainierbar.")
    else:
        for p in model.feature_extractor.parameters():
            p.requires_grad = True

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
    Trainiert das Modell.

    Args:
        trainloader:
            DataLoader mit augmentierten Trainingsdaten.

        valloader:
            DataLoader für Validierung.
            Kann None sein, z.B. beim finalen Training auf trainval.

        model:
            YawDDclassifier.

        epochs:
            Maximale Anzahl Epochen.

        lr:
            Lernrate.

        freeze_backbone:
            0 oder 1.

        device:
            CPU/GPU.

        threshold:
            Klassifikationsschwellwert.

        patience:
            Early-Stopping-Geduld.

        log_dir:
            TensorBoard-Logverzeichnis.

        train_eval_loader:
            Optionaler DataLoader für Trainingsmetriken ohne Augmentation.

        save_path:
            Pfad für bestes state_dict.

        early_stopping:
            True:
                Speichert bestes Modell nach Validation-F1.

            False:
                Kein Early Stopping. Modell wird nach jeder Epoche gespeichert.

    Returns:
        best_f1, best_epoch
    """

    save_dir = os.path.dirname(save_path)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    model = model.to(device)

    apply_freeze_backbone(model, freeze_backbone)

    pos_weight = compute_pos_weight_from_dataset(trainloader.dataset, device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=1e-2
    )

    scheduler = None

    if valloader is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.7,
            patience=5
        )

    # TensorBoard-Modellgraph optional loggen
    # try/except, weil add_graph je nach Torch-Version manchmal empfindlich ist.
    try:
        model.eval()

        with torch.no_grad():
            frames, _ = next(iter(trainloader))
            dummy_input = frames.to(device)
            writer.add_graph(model, dummy_input)

        model.train()

    except Exception as e:
        print(f"TensorBoard add_graph übersprungen: {e}")

    best_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0

        for frames, labels in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            frames = frames.to(device)
            labels = labels.to(device).float().view(-1)

            optimizer.zero_grad()

            logits = model(frames).view(-1)

            loss = criterion(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_loss:.4f}")

        writer.add_scalar("Loss/train", avg_loss, epoch)

        # Trainingsmetriken ohne Augmentation
        train_metrics = None

        if train_eval_loader is not None:
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

            print(f"Train F1: {train_metrics['f1']:.3f}")

        # Validierung
        val_metrics = None

        if valloader is not None:
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

            print(f"Val Accuracy: {val_metrics['accuracy']:.3f}")
            print(f"Val F1      : {val_metrics['f1']:.3f}")
            print(f"Val Recall  : {val_metrics['recall']:.3f}")
            print(f"Val Precision: {val_metrics['precision']:.3f}")

            if scheduler is not None:
                scheduler.step(val_metrics["f1"])

        # Lernrate loggen
        for param_group in optimizer.param_groups:
            current_lr = param_group["lr"]
            writer.add_scalar("LearningRate", current_lr, epoch)
            print(f"Current LR: {current_lr}")

        # Early Stopping anhand echter Validierungsdaten
        if early_stopping and valloader is not None:
            current_f1 = val_metrics["f1"]

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_epoch = epoch
                epochs_no_improve = 0

                torch.save(model.state_dict(), save_path)
                print(f"Neues bestes Modell gespeichert: {save_path}")

            else:
                epochs_no_improve += 1
                print(f"Keine Verbesserung seit {epochs_no_improve} Epoche(n).")

            if epochs_no_improve >= patience:
                print(f"\nEarly stopping nach {epoch + 1} Epochen.")
                break

        else:
            # Finales Training ohne Validation:
            # Modell nach jeder Epoche speichern.
            torch.save(model.state_dict(), save_path)
            best_epoch = epoch

            if train_metrics is not None:
                best_f1 = train_metrics["f1"]

    # Bestes Modell zurückladen, wenn Early Stopping verwendet wurde
    if early_stopping and valloader is not None and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"Bestes Modell aus {save_path} zurückgeladen.")

    writer.close()

    return best_f1, best_epoch
