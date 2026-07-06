import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from torchinfo import summary
from tqdm import tqdm  

from src.evaluation import evaluate



class YawDDclassifier(nn.Module): #Klasse für Neuronales Netz
    def __init__(self, dropout):
        super().__init__()

        # pretrained resnet model
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1]) # keep only the model backbone and remove the final head
        
        # temporal attention pooling
        # Fokus auf die Frames legen, in denen tatsächlich gegähnt wird. Frames, in denen nichts relevantes passiert, werden weniger beachtet.
        self.attn = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # classification head
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
        B, T, C, H, W = x.shape
        
        # frame-wise feature extraction with 2D backbone
        x = x.view(B * T, C, H, W)    # (B*T, C, H, W)
        x = self.feature_extractor(x)           # (B*T, F, 1, 1)
        x = x.view(B, T, -1)                    # (B, T, F)

        # attention pooling over time
        scores = self.attn(x)               # (B, T, 1)
        weights = torch.softmax(scores, dim=1) #Rechnet Scores in Wahrscheinlichkeit um, Summe über Zeit =1
        # Wichtige Frames bekommen hohe Gewichte, unwichtige quasi nahe 0
        pooled = (x * weights).sum(dim=1)   # (B, F)

        # final logits
        logits = self.cls_head(pooled).squeeze(-1)  # (B,)
        return logits
    

#Traniert das Modell
def trainer(trainloader,
            valloader,
            model,
            epochs,
            lr,
            freeze_backbone, 
            device,
            threshold,
            patience=10): #für early stopping kfold
    
    #speichert bestes Ergebnis
    best_f1 = 0
    best_epoch = 0
    epochs_no_improve = 0 #für early stopping kfold







    #Aus Trainingsdaten:
    num_pos = 46 
    num_neg = 132 -46
    pos_weight = num_neg / num_pos
    # objective function is binary cross entropy loss with logits
    criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(pos_weight).to(device) #pos_weigt ergibt sich aus manuell gezählten Labels
    )
    #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0)) #positive Klasse wird stärker gewichtet (bei Klassenungleichgewicht)
    
    # set non-trainable parameters
    # ResNet wird nicht trainiert
    # Backbone wird eingefroren, evt, zu restiktiv, könnte überarbeitet werden.
    """if freeze_backbone:
        for p in model.feature_extractor.parameters():
            p.requires_grad=True #=False #keine Gradienten, keine Updates
            # Modell lernt nur: Attention und Klassifikations-Head
    """

    #Nur letzte Schicht des Backbone unfreezen
    if freeze_backbone:
        for p in model.feature_extractor.parameters():
            p.requires_grad = False

        # letzte Schicht wieder freigeben
        for p in model.feature_extractor[-1].parameters():
            p.requires_grad = True
    

    # Get trainable parameters and hand to optimizer
    # Nur trainierbare Parameter auswählen
    tp = [p for p in model.parameters() if p.requires_grad]
    #L2-Regularisierung: Fügt der Loss-Funktion eine Strafe proportional zum Quadrat der Gewichte des Modells hinzu
    # --> kleinere, besser verteilte Gewichte, reduziert Overfitting
    optimizer = optim.AdamW(tp, lr=lr, weight_decay=1e-2) # AdamW uses weight decay with default 1e-2

    #LR-Schedluer eingefüht:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',        # F1 maximieren
    factor=0.7,        # LR bei Plateau halbieren (0.5)
    patience=5,        # Änderung nach 2 Epochen (2)
    )




   
    
    # summary(model)

    # train loop
    # Wiederholt Training für mehrere Epochen
    for epoch in range(epochs): 

        # init running loss
        # Loss: Differenz zwischen den Vorhersagewerten und den wahren Werten (Labels)
        # --> Verschiedene Loss-Funktionen möglich
        # Running Loss: running loss is the cumulative average loss over a certain number of batches during the training proces
        # --> Stabilere und flüssigere Schätzung der Modell-Performance
        running_loss = 0

        # go through all data
        model.train() #setzt Modell in den Trainingsmodus
        for frames, labels in tqdm(trainloader, desc=f'Epoch {epoch}'): #iteriert über Trainingsdaten
            frames, labels = frames.to(device), labels.to(device) # shift data to device

            # forward + backward pass
            # Vorwärtsdurchlauf, Loss berechnen, Gradienten berechnen
            optimizer.zero_grad()
            logits = model(frames)          
            loss    = criterion(logits, labels)
            loss.backward() 
            # Verhindert exploding Gradient Problem: Gradient könnte unendlich groß werden
            # Exploding gradients occur when gradients grow too large during backpropagation, 
            # leading to unstable weight updates and divergence in loss. 
            # When derivatives or weights are greater than 1, their repeated multiplication 
            # across layers leads to exponential growth       
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping                
            optimizer.step() #Gewichte updaten

            # update running loss
            running_loss += loss.item() #Loss aufsummieren
        
        print(f'  Loss: {running_loss:0.4f}') #Ausgabe

        # evaluate train and validation data
        # Berechne Performance-Metrics
        train_metrics = evaluate(trainloader, model, device, threshold)
        val_metrics = evaluate(valloader, model, device, threshold)


        # Accuracy: Verhältnis zwischen richtigen und falschen Vorhersagen
        # F1:F-score or F-measure is a measure of predictive performance. 
        # It is calculated from the precision and recall of the test, where the precision is the number of true positive 
        # results divided by the number of all samples predicted to be positive, 
        # including those not identified correctly, and the recall is the number of true positive results 
        # divided by the number of all samples that should have been identified as positive.
        # F1c: ???
        print(f"Train Acc: {train_metrics['accuracy']:.3f}   --   Val Acc: {val_metrics['accuracy']:.3f}")
        print(f"Train F1: {train_metrics['f1']:.3f}   --   Val F1c: {val_metrics['f1']:.3f}")

        

         #Für LR-Scheduler:
        scheduler.step(val_metrics['f1']) 
        #Aktuelle Lernrate ausgeben:
        for param_group in optimizer.param_groups:
            print("Current LR:", param_group['lr'])
            

        # Save best model checkpoint
        # Speichert bestes Ergebnis und gibt dieses aus
        # Speichert NICHT das beste Modell --> optimierbar?
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch

            epochs_no_improve = 0  # für early stopping kfold

            # Modell speichern
            torch.save(model.state_dict(), "best_model.pt")
        
        else:
            epochs_no_improve += 1 # für early stopping kfold
        
        if epochs_no_improve >= patience: # für early stopping kfold
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    return best_f1, best_epoch
