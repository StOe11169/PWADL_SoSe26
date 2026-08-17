import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights # ResNet18-Modell mit pretrained ImageNet-Gewichten
from torchinfo import summary # Für die Modellzusammenfassung (optional)
from tqdm import tqdm  # Fortschrittsbalken für das Training
from src.evaluation import evaluate # Eigene Evaluierungsfunktion
from torch.utils.tensorboard import SummaryWriter # TensorBoard-Logging für Metriken und Modellgraph



class YawDDclassifier(nn.Module): # Neuronales Netz für die Müdigkeitserkennung (Gähnen vs. Kein Gähnen)
    """
    Hybrides Deep-Learning-Modell bestehend aus:
    - ResNet18-Backbone (Feature-Extraktion)
    - Temporaler Attention (Frame-Selektion)
    - Klassifikations-Head (Binäre Entscheidung)
    """
    def __init__(self, dropout):
        """
        Initialisiert das Modell mit den gegebenen Hyperparametern.

        Args:
            dropout (float): Dropout-Rate für Regularisierung (0.2-0.6)
        """
        super().__init__()

        # ===== 1. Feature-Extraction: ResNet18-Backbone =====
        # Pretrained ResNet18 lädt ImageNet-Gewichte für robuste Feature-Extraktion
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Backbone ohne die finale Fully-Connected-Layer verwenden (nur konvolutionelle Features)
        # children() gibt alle Schichten zurück,[:-1] entfernt die letzte Schicht (fc)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1]) 
        
        # ===== 2. Temporale Attention (Frame-Selektion) =====
        # Attention-Mechanismus, um relevante Frames (z.B. Gähnen-Phasen) zu gewichten
        # backbone.fc.in_features gibt die Feature-Dimension nach ResNet (512)
        self.attn = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 128), # Projektion auf 128-dim Hidden-Layer
            nn.Tanh(), # Tanh-Aktivierung für nicht-lineare Transformation
            nn.Linear(128, 1), # Projektion auf Score pro Frame
        )

        # ===== 3. Klassifikations-Head (Binäre Entscheidung) =====
        # Fully-Connected-Netzwerk für die finale Klassifikation
        self.cls_head = nn.Sequential(
            # Erste FC-Layer mit BatchNorm und ReLU
            nn.Linear(backbone.fc.in_features, 256),
            nn.BatchNorm1d(256), # Normalisierung für stabileres Training
            nn.ReLU(inplace=True), # ReLU-Aktivierung
            nn.Dropout(dropout), # Dropout für Regularisierung
            # Zweite FC-Layer
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # Finale FC-Layer für binäre Klassifikation (1 Output)
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """
        Vorwärtsdurchlauf des Modells.

        Args:
            x (torch.Tensor): Eingabetensor mit Form (Batch, Zeit, Kanäle, Höhe, Breite)

        Returns:
            torch.Tensor: Logits für binäre Klassifikation (Batch-Größe)
        """
        B, T, C, H, W = x.shape # Batch-Größe, Anzahl Frames, Kanäle, Höhe, Breite
        
        # ===== 1. Feature-Extraktion mit ResNet18 =====
        # Reshape für 2D-CNN: (B,T,C,H,W) → (B*T,C,H,W)
        x = x.view(B * T, C, H, W)    # (B*T, C, H, W)
        # Features mit ResNet extrahieren
        x = self.feature_extractor(x)           # Output: (B*T, 512, 1, 1)
        # Reshape zurück zu (B,T,512) für temporale Analyse
        x = x.view(B, T, -1)

        # ===== 2. Temporale Attention =====
        # Attention-Scores berechnen: (B,T,512) → (B,T,1)
        scores = self.attn(x)

        # Softmax über Zeitachse für Normalisierung der Scores
        # weights gibt an, wie wichtig jeder Frame für die Klassifikation ist
        weights = torch.softmax(scores, dim=1) 
        # Gewichtete Summierung der Features: (B,T,512) ⊙ (B,T,1) → (B,512)
        # ⊙ = Elementweise Multiplikation
        pooled = (x * weights).sum(dim=1)

        # ===== 3. Klassifikations-Head =====
        # Finale Logits berechnen: (B,512) → (B,1)
        logits = self.cls_head(pooled).squeeze(-1) # squeeze(-1) entfernt letzte Dimension
        return logits # Roh-Scores für binäre Klassifikation
    

# Funktion zum Trainieren des Modells
def trainer(trainloader,
            valloader,
            model,
            epochs,
            lr,
            freeze_backbone, 
            device,
            threshold,
            patience=10, # Anzahl der Epochen ohne Verbesserung, bevor Early Stopping ausgelöst wird
            log_dir="runs"):  # Verzeichnis für TensorBoard-Logs
    """
    Trainiert das Modell mit gegebenen Hyperparametern und evaluiert auf Validierungsdaten.

    Args:
        trainloader (DataLoader): Dataloader für Trainingsdaten
        valloader (DataLoader): Dataloader für Validierungsdaten
        model (YawDDclassifier): Zu trainierendes Modell
        epochs (int): Anzahl der Trainingsepochen
        lr (float): Lernrate für den Optimizer
        freeze_backbone (int): 0=Backbone trainieren, 1=Backbone einfrieren
        device (torch.device): Gerät (CPU/GPU)
        threshold (float): Entscheidungsgrenze für binäre Klassifikation
        patience (int): Patience für Early Stopping
        log_dir (str): Verzeichnis für TensorBoard-Logs

    Returns:
        tuple: (best_f1, best_epoch) - Beste F1-Score und Epoche
    """

    # ===== Initialisierung =====
    # TensorBoard-Logger initialisieren für Metriken-Visualisierung
    writer = SummaryWriter(log_dir=log_dir)
    
    # Beste Metriken initialisieren
    best_f1 = 0 # Beste F1-Score
    best_epoch = 0 # Epoche, in der bester F1-Score erreicht wurde
    epochs_no_improve = 0 # Zähler für Epochen ohne Verbesserung (für Early Stopping)


    # ===== Loss-Funktion und Optimizer =====
    # Klassenungleichgewicht berechnen: 86 Videos ohne Gähnen / 46 mit Gähnen = 1.87 NOCH ÄNDERN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    num_pos = 46 
    num_neg = 132 -46
    pos_weight = num_neg / num_pos # Strafe für False Negatives (übersehene Gähnen)

    # Binary Cross Entropy Loss mit Positiv-Gewichtung für bessere Klassifikation seltener Klasse
    criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(pos_weight).to(device) # Gewichtung für seltene Klasse (Gähnen)
    )

    # ===== Backbone einfrieren (falls freeze_backbone=1) =====
    if freeze_backbone:
        # Alle Parameter des Backbones auf nicht-trainierbar setzen
        for p in model.feature_extractor.parameters():
            p.requires_grad = False

        # Letzte Schicht des Backbones wieder freigeben für Training
        # [-1] greift auf den letzten ResNet-Block zu
        for p in model.feature_extractor[-1].parameters():
            p.requires_grad = True
    

    # ===== Trainierbare Parameter auswählen =====
    # Nur Parameter, die trainierbar sind (requires_grad=True), werden an den Optimizer übergeben
    tp = [p for p in model.parameters() if p.requires_grad]
    # AdamW-Optimizer mit L2-Regularisierung (weight_decay=1e-2)
    optimizer = optim.AdamW(tp, lr=lr, weight_decay=1e-2) # AdamW uses weight decay with default 1e-2

    # ===== Lernraten-Scheduling =====
    # Reduziert Lernrate, wenn F1-Score über mehrere Epochen stagniert
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',        # Maximiere F1-Score
    factor=0.7,        # Lernrate wird um 30% reduziert
    patience=5,        # 5 Epochen ohne Verbesserung → Trigger
    )


    # ===== Modellgraph für TensorBoard =====
    # Erstelle einen Dummy-Input für das Logging des Modellgraphen
    # Nimmt das erste Batch aus dem trainloader für realistische Daten
    frames, _ = next(iter(trainloader))  
    dummy_input = frames.to(device)[:1] # Nur erstes Sample für schnelleres Logging
    writer.add_graph(model, dummy_input) # Modellgraph in TensorBoard speichern


    # ===== Trainingsloop =====
    for epoch in range(epochs): 
        # Initialisiere laufenden Loss für diese Epoche
        running_loss = 0

        # Modell in Trainingsmodus setzen (aktiviert Dropout, BatchNorm in Trainingsmodus)
        model.train()
        # Iteriere über alle Batches im Trainings-Dataloader
        for frames, labels in tqdm(trainloader, desc=f'Epoch {epoch}'):
            # Daten auf richtiges Gerät verschieben (CPU/GPU)
            frames, labels = frames.to(device), labels.to(device)

            # ===== Vorwärtsdurchlauf =====
            # Gradienten zurücksetzen
            optimizer.zero_grad()
            # Logits berechnen (Rohwerte ohne Aktivierung)
            logits = model(frames)
            # Loss berechnen       
            loss    = criterion(logits, labels)

            # ===== Rückwärtsdurchlauf =====
            loss.backward() 

            # ===== Gradient Clipping =====
            # Verhindert explodierende Gradienten      
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Parameter aktualisieren               
            optimizer.step()

            # Laufenden Loss aktualisieren
            running_loss += loss.item()
        
        print(f'  Loss: {running_loss:0.4f}') #Ausgabe

        # Evaluierung auf Trainings- und Validierungsdaten
        train_metrics = evaluate(trainloader, model, device, threshold, writer=writer, epoch=epoch)
        val_metrics = evaluate(valloader, model, device, threshold, writer=writer, epoch=epoch)


        # ===== TENSORBOARD-LOGGING (einmal pro Epoche) =====
        writer.add_scalar('Loss/train', running_loss / len(trainloader), epoch)  # Durchschnittlicher Loss
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('F1/train', train_metrics['f1'], epoch)

        # ===== Lernrate loggen =====
        for param_group in optimizer.param_groups:
            writer.add_scalar('LearningRate', param_group['lr'], epoch)

        # ===== Ausgabe der Metriken =====
        print(f"Train Acc: {train_metrics['accuracy']:.3f}   --   Val Acc: {val_metrics['accuracy']:.3f}")
        print(f"Train F1: {train_metrics['f1']:.3f}   --   Val F1c: {val_metrics['f1']:.3f}")


        

        # ===== Lernraten-Scheduling =====
        # Reduziert Lernrate, wenn F1-Score über mehrere Epochen stagniert
        scheduler.step(val_metrics['f1']) 



        #Aktuelle Lernrate ausgeben:
        for param_group in optimizer.param_groups:
            print("Current LR:", param_group['lr'])
            

        # ===== Early Stopping =====
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1'] # Beste F1-Score aktualisieren
            best_epoch = epoch # Beste Epoche speichern
            epochs_no_improve = 0  # Zähler zurücksetzen

            # Beste Modell-Gewichte speichern
            torch.save(model.state_dict(), "best_model.pt")
        
        else:
            epochs_no_improve += 1 # Zähler erhöhen

        # Early Stopping auslösen, wenn keine Verbesserung über 'patience' Epochen
        if epochs_no_improve >= patience: 
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # TensorBoard-Logger schließen
    writer.close() 
    return best_f1, best_epoch # Beste F1-Score und Epoche zurückgeben
