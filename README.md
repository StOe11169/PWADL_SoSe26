"# PWADL_SoSe26"
Problembeschreibung:
Problemstellung: Ziel des Projektes ist die Entwicklung eines automatisierten Müdigkeitserkennungssystems für Videosequenzen, das Gähnen als Indikator für Ermüdung zuverlässig identifiziert und klassifiziert.

Modell-Architektur: ResNet18 + Temporale Attention
Feature-Extraction (ResNet18-Backbone): Extrahiert konvolutionelle Featrures aus Einzelbildern der Videosequenzen. Die Eingabe erfolgt in Form von Batches an Frames mit der Form (B, T, C, H, W) (Batch, Zeit, Kanäle, Höhe, Breite). Der pretrained ResNet18 (ImageNet) wird als Backbone genutzt. Die letzten Fully-Connected-Layer werden entfernt (nn.Sequential(*list(backbone.children())[:-1])), sodass nur die konvolutionellen Feature-Maps übrigbleiben. Optional kann der Backbone eingefroren werden (freeze_backbone=1), um nur die letzten Schichten zu trainieren und Overfitting zu reduzieren. Als Ausgabe erhält man 512-dimensionale Feature-Vektoren pro Frame mit Form (B, T, 512). 

Temporale Attention (Frame Selection): Frames werden dynamisch gewichtet, um relevante Müdigkeitsphasen (Gähnen) zu priorisieren. Dazu werden Attention-Scores berechnet, diese normalisiert und summiert, um eine komprimierte Repräsentation der Videosequenz zu erhalten. Als Ausgabe erhält man eine Gewichtete Feature-Matrix, wodurch irrelevante Frames unterdrückt werden

Binärer Klassifikations-Head: Klassifiziert die Gewichteten Features als „Gähnen (1)“ oder „Kein Gähnen (0)“. Die 512-dimensionale Feature-Vektor wird durch ein Fully-Connected-Netzwerk mit BatchNorm, ReLU und Dropout geleitet. Der Logit wird mit sigmoid aktiviert und mit einem Schwellenwert (threshold) verglichen, um die finale Klasse zu bestimmen.

Mathematische Beschreibung: 
Feature-Extraction:
Input Definition: 
•	Form: X ∈ ℝ^(B × T × C × H × W)
o	B: Batch-Größe (Anzahl der Videosequenzen pro Batch)
o	T: Anzahl der Frames pro Sequenz (z. B. T = 32)
o	C: Kanäle (RGB → C = 3)
o	H, W: Höhe und Breite der Frames (nach CenterCrop → H = W = 224)
•	Ground Truth (Labels):
o	Y ∈ {0,1}^(B): Binäres Label pro Sequenz (1 = Gähnen, 0 = Kein Gähnen)

Herleitung:
Schritt 1: Reshape für 2D-CNN
Der Input wird in eine Batch-weise Frame-Struktur umgewandelt: 
Formel: X_reshaped = reshape(X, (B·T, C, H, W))
Input: X ∈ ℝ^(B × T × C × H × W)
Output: X_reshaped ∈ ℝ^((B·T) × C × H × W)

Schritt 2: Feature-Extraktion mit ResNet18
ResNet18 extrahiert Features pro Frame: Formel: F = ResNet18(X_reshaped)
Output: F ∈ ℝ^((B·T) × 512)

Schritt 3: Reshape zurück zu Videosequenz
Formel: F_sequence = reshape(F, (B, T, 512))
Output: F_sequence ∈ ℝ^(B × T × 512)

Temporale Attention (Frame-Selektion):
Input-Definition
•	Form: F_sequence ∈ ℝ^(B × T × 512)
o	B: Batch-Größe
o	T: Anzahl der Frames pro Sequenz
o	512: Feature-Dimension pro Frame
Herleitung:
Schritt 1: Lineare Projektion der Features
Formel: scores = W₂ · tanh(W₁ · F_sequence + b₁) + b₂
Gewichte:
•	W₁ ∈ ℝ^(512 × 128)
•	W₂ ∈ ℝ^(128 × 1)
•	Bias: b₁ ∈ ℝ^(128), b₂ ∈ ℝ
Output: scores ∈ ℝ^(B × T × 1)

Schritt 2: Softmax-Normalisierung
Formel: weights = softmax(scores, dim=1)
Eigenschaft: ∑(t=1 bis T) weights_(b,t) = 1
Output: weights ∈ ℝ^(B × T × 1)

Schritt 3: Gewichtete Summierung
Formel: pooled = ∑(t=1 bis T) (F_sequence,b,t ⊙ weights_b,t)
Output: pooled ∈ ℝ^(B × 512)

Klassifikations-Head (Binäre Entscheidung)
	Input-Definition
•	Form: pooled ∈ ℝ^(B × 512)

Herleitung:
Schritt 1: Fully-Connected-Netzwerk
Formel 1: Z₁ = ReLU(BatchNorm(Linear_(512→256)(pooled) · W₁ + b₁))
Formel 2: Z₂ = ReLU(BatchNorm(Linear_(256→128)(Z₁) · W₂ + b₂))
Formel 3: logits = Linear_(128→1)(Z₂) · W₃ + b₃
Gewichte:
•	W₁ ∈ ℝ^(512 × 256), W₂ ∈ ℝ^(256 × 128), W₃ ∈ ℝ^(128 × 1)
•	Dropout: p ∈ [0.2, 0.6]

Schritt 2: Sigmoid & Thresholding
Formel 1: probs = σ(logits) = 1 / (1 + e^(-logits))
Formel 2: Ŷ = 𝟙(probs > θ) mit θ ∈ [0.25, 0.35]

Gesamtmodell:
Forward-Pass
Formel: Ŷ = Model(X, W, θ) = 𝟙(σ(Linear_Head(Attention(ResNet18(X)))) > θ)
Loss: L(W, θ) = BCEWithLogitsLoss(logits, Y) + λ · ||W||²

Notation:
•	ResNet18(·): Pretrained ResNet18 (ohne finale FC-Layer)
•	tanh(·): Hyperbolischer Tangens
•	softmax(·, dim=1): Normalisierung über Zeitachse
•	⊙: Elementweise Multiplikation
•	σ(·): Sigmoid-Funktion
•	𝟙(·): Indikatorfunktion
•	||W||²: L2-Norm der Gewichte (Regularisierung)


Datensatz:
BISHER NICHT ALLES VERWENDET, NOCH ANPASSEN
Als Grundlage für das Projekt dient der Datensatz „YawDD: Yawning Detection Dataset“, erhältlich unter YawDD: Yawning Detection Dataset | IEEE DataPort. Konkret werden daraus die alle Videos der Unterkategorie Mirror verwendet. Des Weiteren wurden noch selbsterstellte Videos von fünf männlichen Personen hinzugefügt, die dem Datensatz über die IDs 48 bis 52 angehängt wurden. Der gesamte Datenumfang ergibt sich daraus zu: Female IDs 1-43 (156 Elemente) + Male IDs 1-47 (164 Elemente) + Eigene Daten Male IDs 48-52 (14 Elemente). Die Videos haben je nach Inhalt die Bezeichnungen NoGLasses, Glasses, SunGlasses und die Label Talking, Yawning oder Normal. Alle Videos sind zwischen 15 und 30 Sekunden lang und haben eine Auflösung von 640x480 Pixeln bei 30 Bildern/Sekunde
Verarbeitung: Eigens aufgenommene Videos wurden von 4k60fps auf das gleiche Format wie YawDD konvertiert.

Anleitung herunterladen und Preprocessing?

Anleitung Visualisierung?

Die Trainingsdaten werden mit stochastischen Augmentierungen angereichert, um das Modell robuster gegenüber Variationen zu machen. Die Augmentierungen (RandomHorizontalFlip, RandomRotation, ColorJitter) werden nur auf die Trainingsdaten angewendet (train=True), während Validierungs- und Testdaten unverändert bleiben (train=False). Dadurch lernt das Modell, Gähnen auch bei unterschiedlichen Beleuchtungen, Perspektiven oder Gesichtsausdrücken zu erkennen, ohne die echte Verteilung der Validierungs-/Testdaten zu verzerren.


Code und Anweisungen zum Ausführen des Repositorys
Benötigte Bibliotheken und Pakete:
Dieses Projekt verwendet Python 3.13.13
Genaue Versionen und Abhängigkeiten können der requirements.txt entnommen werden. Die wichtigsten Komponenten sind:
Torch: Deep-Learning-Framework (PyTorch) für Modell-Training & Inference 
Torchvision: Bildverarbeitung (z. B. transforms.Compose, ResNet18) 
Optuna: Hyperparameter-Optimierung 
Tensorboard:  Visualisierung von Metriken (Loss, Accuracy, Modellgraphen) 
Torchcodec: Video-Decoding (für MP4/AVI/MOV-Dateien) 
Tqdm: Fortschrittsbalken für Training/Evaluation 
scikit-learn: Metriken-Berechnung (Accuracy, F1-Score, etc.) 
pandas: Datenverwaltung (Pfad-Handling, Label-Extraktion) 
numpy: Numerische Berechnungen (z. B. torch.linspace für Frame-Selektion) 
matplotlib: Optional: Visualisierung von Attention-Heatmaps oder Frame-Beispielen

Virtuelle Umgebung einreichten:
1: python -m venv .venv
2: .\.venv\Scripts\activate
3: pip install -r requirements.txt

Projektstruktur und Daten-Setup:
PWADL_SoSe26/
|--main.py 				#Optuna + Training
|--README.md 			#Projektdokumentation
|--requiremetns.txt			 #Setup-Informationen
|--tester.py 				#Testen auf anderen Daten
|--runs/				#Enthält nach dem Training Tensorboard-Daten
|--src/
|--training.py 			#Modell-Architektur
|--evaluation.py 		#Metrikenberechnung
|--data.py 			#Datenaufbereitung
|--utils.py 			#Reproduzierbarkeit sichern
|--data/ 				#Datensatz-Ordner
	|--train/ Videos.avi 		#Trainingsvideos
	|--val/ Videos.avi 		#Validierungsvideos
	|--test/ Videos.avi 		#Testvideos



Ausführen des Projektes:
Hyperparameter-Optimierung (Optuna): python main.py --num_frames 32 --epochs 20 --n_trials 5
Finale Training mit besten Parametern: python main.py --num_frames 32 --epochs 20 --n_trials 1
Tensorboard starten: tensorboard --logdir runs/
Modell testen: python tester.py --num_frames 32


Training und Hyperparameter-Konfiguration:
Das Training verfolgt einen zweistufigen Ansatz mit Optuna-Hyperparameter-Optimierung und finalem Training mit den besten Parametern
Übersicht Hyperparameter:
Hyperparameter	Wertebereich	Beschreibung	Typ
batch_size	[4, 8]	Anzahl der Videosequenzen pro Batch	Integer
freeze_backbone	[0, 1]	0 = Backbone trainiert komplett, 1 = nur letzte Schicht trainiert (vermindert Overfitting).	Boolean
lr	1e-5 bis 2e-4	Lernrate (logarithmisch optimiert für stabile Konvergenz).	Float
dropout	0.2 bis 0.6	Dropout-Rate im Klassifikations-Head (Regularisierung gegen Overfitting).	Float
threshold	0.25 bis 0.35	Schwellenwert für Sigmoid-Aktivierung (optimiert F1-Score).	Float
num_frames	4 bis 64	Anzahl der Frames pro Videosequenz	Integer
epochs	3 bis 30	Anzahl der Trainingsepochen (Early Stopping stoppt ggf. früher).	Integer
n_trials	1 bis 5	Anzahl der Optuna-Trials 	Integer

Pro Optuna Trial werden durch K-Fold Cross-Validation 5 Folds durchlaufen.
Beste Hyperparameter werden in study.best_params gespeichert.
Im finalen Training wird das Modell mit den besten Parametern noch einmal auf den gesamten Trainings- und Validierungsdaten trainiert.
Eingebaute Features im Training:
•	K-Fold Cross-Validation: Verhindert Overfitting durch robuste Validierung. 5 Folds, jedes Video wird einmal im Validation-Set verwendet. Pro Trial werden 5 Modelle trainiert und der durchschnittliche F1-Score daraus bestimmt die Qualität des Hyperparametersets.
•	Early Stopping: Stoppt Training, falls sich F1-Score 10 Epochen nicht verbessert. Verhindert unnötige Rechenzeit.
•	Positiv-Gewichtung im Loss: Klassenungleichgewicht (86 Videos ohne Gähnen vs. 46 mit Gähnen) → pos_weight = 2.87
•	Lernraten-Scheduling: Reduziert Lernrate, wenn F1-Score plateauiert (ReduceLROnPlateau). Parameter:
o	factor=0.7 → Lernrate wird um 30% reduziert.
o	patience=5 → 5 Epochen ohne Verbesserung → Trigger.
•	Gradient Clipping: verhindert explodierende Gradienten
•	TensorBoard-Integration: Visualisierung von Loss, Metriken und Modellgraphen

Evaluationscode:
Die evaluate()-Funktion misst die Performance des Modells auf einem gegebenen Datensatz (Train, Validation oder Test) und gibt vier zentrale Metriken zurück:
•	Accuracy (Gesamt-Genauigkeit) = (TP + TN) / (TP + TN + FP + FN)
•	Precision (Anteil korrekter Gähnen-Vorhersagen) = TP / (TP + FP)
•	Recall (Anteil erkannter Gähnen-Fälle) = TP / (TP + FN)
•	F1-Score (Balance zwischen Precision und Recall) = 2 * (Precision * Recall) / (Precision + Recall)
Zusätzlich kann sie die Metriken in TensorBoard loggen (falls writer und epoch übergeben werden), um den Fortschritt während des Trainings zu visualisieren.
Dazu muss das Modell über model.eval() in den Evaluationsmodus geschaltet werden, was dropout und BatchNorm deaktiviert.

Ergebnisse und Diskussion:
Laufzeit und Ressourcen:
Core Ultra 9 285H/32GB RAM (CPU-Training): Beste Ergebnisse mit num_frames:32, Epochs: 25, Trials: 5 
	Trainingszeit 19:18:25 (hh:mm:ss), Test Acc: 0.900; Test F1:  0.882
Beste Parameter:
•	batch_size = 8
•	freeze_backbone =  0
•	lr = 0.00013121430541763315
•	dropout = 0.4
•	threshold = 0.30596314369198296

Vergleich mit: Ryzen 9 5900HS, RTX3050Laptop, 16GB RAM (GPU-Training): Langsamer, da RAM-Limitiert.

Aufgrund des Ungleichgewichts der Klassen wurde der F1-Score als Hauptmetrik gewählt, da sonst das Modell immer eine Klasse vorhersage könnte, um eine hohe Accuracy zu erhalten


VISUALISIERUNG

VERGLEICH MIT ANDEREN MODELLEN

BESONDERHEITEN / SCHWÄCHEN DES MODELLS
Was gut, was schwierig?
