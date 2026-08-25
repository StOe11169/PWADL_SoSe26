# Müdigkeitserkennung durch Gähn-Detektion in Fahrervideos
## Projektarbeit im Modul PWADL, SoSe 2026
In dieser Projektarbeit wird ein Deep-Learning-Modell zur automatischen Erkennung von Gähnen in Fahrervideos entwickelt. Gähnen wird dabei als mögliches visuelles Merkmal für Müdigkeit am Steuer betrachtet. Als Datengrundlage wird der YawDD-Datensatz verwendet, der durch eigene Videos im gleichen Format ergänzt wurde. Das Modell klassifiziert Videosequenzen binär in die Klassen yawning und non-yawning.
Die entwickelte Pipeline umfasst das Einlesen von Videodaten aus einem Sammelordner, die reproduzierbare Aufteilung aller Daten in Trainings-, Validierungs- und Testdaten, eine framebasierte Vorverarbeitung, ein neuronales Netz mit ResNet18-Backbone und temporaler Attention sowie Training, Hyperparameteroptimierung, Logging und Evaluation. Die finale Bewertung erfolgt auf einem unabhängigen Testsplit.

## Inhaltsverzeichnis

1. [Problembeschreibung](#1-problembeschreibung)
2. [Modellarchitektur](#2-modellarchitektur)
3. [Loss-Funktion und Optimierung](#3-loss-funktion-und-optimierung)
4. [Datensatz](#4-datensatz)
5. [Datenvorverarbeitung](#5-datenvorverarbeitung)
6. [Implementierung](#6-implementierung)
7. [Training](#7-training)
8. [Evaluation](#8-evaluation)
9. [Logging und Visualisierung](#9-logging-und-visualisierung)
10. [Ergebnisse](#10-ergebnisse)
11. [Diskussion](#11-diskussion)
12. [Vergleich mit anderen Ansätzen](#12-vergleich-mit-anderen-ansätzen)
13. [Reproduzierbarkeit](#13-reproduzierbarkeit)
14. [Hinweise zur Konsolenausgabe](#14-hinweise-zur-konsolenausgabe)
15. [Hilfsskript summarize_trials.py](#15-hilfsskript-summarize_trialspy)
16. [Hilfsskript final_train_only.py](#16-hilfsskript-final_train_onlypy)
17. [Quellen](#17-quellen)
18. [Anhang](#18-anhang)

## 1. Problembeschreibung
### 1.1 Motivation
Müdigkeit am Steuer stellt ein sicherheitsrelevantes Risiko dar, da sie die Reaktionsfähigkeit und Aufmerksamkeit der fahrenden Person beeinträchtigen kann. Viele Automobilhersteller entwickeln daher seit Jahren eigene Systeme, um Müdigkeit und Erschöpfung zu erkennen. Ein mögliches visuelles Anzeichen für Müdigkeit ist Gähnen. Ziel dieses Projekts ist es daher, ein Modell zu entwickeln, das anhand kurzer Videosequenzen erkennt, ob eine Person gähnt oder nicht. Dazu wird die Aufgabe als binäres Klassifikationsproblem umgesetzt. Für jede Videosequenz wird vorhergesagt, ob sie zur Klasse yawning oder zur Klasse non-yawning gehört.

### 1.2 Zielsetzung
Ziel der Arbeit ist die Entwicklung und Evaluation einer Deep-Learning-Pipeline zur Gähn-Erkennung in Fahrervideos. Die Pipeline soll folgende Anforderungen erfüllen:

1. Einlesen aller Videodaten aus einem gemeinsamen Datenordner. 
2. Automatische Erzeugung und Nutzung einer reproduzierbaren Split-Datei. 
3. Gruppierte Aufteilung nach Personen- / Video-ID, um Datenleckage zu reduzieren. 
4. Extraktion einer festen Anzahl gleichmäßig verteilter Frames pro Video. 
5. Training eines geeigneten Modells für Videodaten. 
6. Hyperparameteroptimierung mittels Optuna. 
7. Evaluation auf einem unabhängigen Testsplit. 
8. Logging von Trainings- und Evaluationsmetriken mit TensorBoard. 
9. Speichern und erneutes Laden des finalen Modells.

### 1.3 Formale Problemdefinition
Ein aus einem Video extrahierter Clip wird als Sequenz von Frames beschrieben: \
X = {x₁, x₂, ..., x_T} \
Dabei bezeichnet T die Anzahl der verwendeten Frames pro Videosequenz. \
Die Zielvariable y ist binär definiert: \
y ∈ {0, 1} \
Dabei gilt: \
y = 1 für die Klasse yawning \
y = 0 für die Klasse non-yawning \
Das Modell berechnet aus der Eingabesequenz einen Logit z. Dieser Logit wird anschließend mit der Sigmoid-Funktion in eine Wahrscheinlichkeit p überführt: \
p = σ(z) = 1 / (1 + e^(-z)) \
Die finale Klassifikation erfolgt über einen Schwellwert τ: \
ŷ = 1, falls p > τ \
ŷ = 0, falls p ≤ τ \
Der Schwellwert τ wird im Rahmen der Hyperparameteroptimierung bestimmt. 

## 2. Modellarchitektur
### 2.1 Überblick
Das verwendete Modell besteht aus drei Hauptkomponenten:

1.	ResNet18-Backbone zur Feature-Extraktion pro Frame.
2.	Temporaler Attention-Mechanismus zur Gewichtung relevanter Frames. 
3.	Fully-Connected-Klassifikationskopf zur binären Klassifikation. 

Die Eingabe des Modells besitzt die Form: \
(B, T, C, H, W) \
Dabei gilt: 
| Symbol | Bedeutung                            |
|--------|--------------------------------------|
| B	     | Batchgröße                           |
| T	     | Anzahl der Frames pro Videoclip      |
| C	     | Anzahl der Farbkanäle, hier C = 3    |
| H	     | Bildhöhe, hier H = 224               |
| W	     | Bildbreite, hier W = 224             |

### 2.2 ResNet18-Backbone
Für jeden Frame wird ein Featurevektor mit einem auf ImageNet vortrainierten ResNet18 extrahiert. Die finale Fully-Connected-Schicht des ResNet18 wird entfernt. Dadurch erzeugt der Backbone pro Frame einen Featurevektor der Dimension 512. \
Die Eingabesequenz hat zunächst die Form: \
X ∈ R^(B × T × 3 × 224 × 224) \
Für die Verarbeitung durch das 2D-CNN wird sie umgeformt zu: \
X' ∈ R^((B · T) × 3 × 224 × 224) \
Nach der Feature-Extraktion wird die Ausgabe wieder als Sequenz dargestellt: \
F ∈ R^(B × T × 512) \
Der Vorteil dieses Ansatzes liegt darin, dass ein vortrainiertes Bildmodell genutzt werden kann. Dadurch müssen visuelle Grundmerkmale wie Kanten, Texturen oder einfache Objektstrukturen nicht vollständig neu gelernt werden.

### 2.3 Temporale Attention
Die verwendeten Videos enthalten nur verhältnismäßig wenige Frames, in denen tatsächlich gegähnt wird. Daher werden die Frame-Features nicht einfach gemittelt. Stattdessen verwendet das Modell einen Attention-Mechanismus, der jedem Frame ein Gewicht zuweist. \
Für jedes Frame-Feature f_t wird zunächst ein Attention-Score berechnet: \
s_t = W₂ · tanh(W₁ f_t + b₁) + b₂ \
Anschließend werden die Scores über die Zeitachse normalisiert: \
α_t = exp(s_t) / Σ exp(s_k) \
Die gewichtete Clip-Repräsentation ergibt sich dann zu: \
f_clip = Σ α_t f_t \
Frames, die für die Gähn-Erkennung besonders relevant sind, können dadurch stärker in die finale Entscheidung einfließen.

### 2.4 Klassifikationskopf
Der aggregierte Featurevektor f_clip wird durch einen Fully-Connected-Klassifikationskopf verarbeitet. Die Struktur lautet:
| Schicht	        | Funktion                       |
|-------------------|--------------------------------|
| Linear(512 → 256)	| Reduktion der Featuredimension |
| BatchNorm1d	    | Stabilisierung des Trainings   |
| ReLU	            | Nichtlineare Aktivierung       |
| Dropout	        | Regularisierung                |
| Linear(256 → 128)	| Weitere Merkmalsverdichtung    |
| BatchNorm1d	    | Stabilisierung des Trainings   |
| ReLU	            | Nichtlineare Aktivierung       |
| Dropout	        | Regularisierung                |
| Linear(128 → 1)	| Ausgabe eines Logits           |
Der finale Output ist ein einzelner Logit für die binäre Klassifikation.

## 3. Loss-Funktion und Optimierung
### 3.1 Loss-Funktion
Für das Training wird BCEWithLogitsLoss verwendet. Diese Loss-Funktion kombiniert die Sigmoid-Aktivierung und die Binary Cross Entropy in einer numerisch stabilen Funktion.
Da die Klassen im Datensatz nicht exakt gleich häufig auftreten, wird eine Positiv-Gewichtung verwendet, die dynamisch aus dem jeweiligen Trainingssplit berechnet wird: \
pos_weight = N_neg / N_pos \
Dabei gilt:
| Symbol | Bedeutung                                                 |
|--------|-----------------------------------------------------------|
| N_neg	 | Anzahl der non-yawning-Videos im aktuellen Trainingssplit |
| N_pos	 | Anzahl der yawning-Videos im aktuellen Trainingssplit     |
Durch diese Gewichtung werden Fehler bei positiven Beispielen stärker berücksichtigt. Dies ist sinnvoll, da die positive Klasse yawning im Datensatz seltener auftreten kann.

### 3.2 Optimierer
Als Optimierer wird AdamW verwendet. AdamW kombiniert adaptive Lernraten mit Weight Decay und trägt dadurch zur Stabilisierung und Regularisierung des Trainings bei. Die Lernrate wird im Rahmen der Hyperparameteroptimierung durch Optuna bestimmt.

### 3.3 Lernraten-Scheduling
Während der Cross-Validation wird ein ReduceLROnPlateau-Scheduler eingesetzt. Dieser überwacht den Validation-F1-Score. Wenn sich der Validation-F1-Score über mehrere Epochen nicht verbessert, wird die Lernrate reduziert.
Die verwendeten Einstellungen sind:
| Parameter	| Wert |
|-----------|------|
| mode	    | max  |
| factor	| 0.7  |
| patience	| 5    |
Damit wird die Lernrate reduziert, wenn der F1-Score auf den Validierungsdaten stagniert.
 ![Verlauf der Lernrate im Training](/pictures/LR_Training.png)
Abbildung 1: Verlauf der Lernraten im Training über verschiedene Trials und Folds

### 3.4 Gradient Clipping
Nach der Backpropagation wird Gradient Clipping verwendet:
```
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Dadurch werden sehr große Gradienten begrenzt. Dies kann das Training stabilisieren und insbesondere bei kleinen Batchgrößen oder komplexeren Modellteilen wie Attention hilfreich sein.

## 4. Datensatz
### 4.1 Datenbasis
Als Grundlage wird das Yawning Detection Dataset, kurz YawDD verwendet. Der Datensatz ist über IEEE DataPort verfügbar und enthält Videos von Fahrerinnen und Fahrern in unterschiedlichen Zuständen, darunter Gähnen und Nicht-Gähnen. 
Zusätzlich wurden eigene Videos aufgenommen, die dem YawDD-Format angeglichen und über fortlaufende IDs an den Datensatz angehängt wurden.
Der YawDD-Datensatz muss aufgrund der Lizenzbedingungen separat von der offiziellen Quelle bezogen werden.
Alle Videos liegen gemeinsam im Ordner data/videos/. Das erwartete Dateinamenformat lautet: \ ID-info_labels-activity, \ wobei Videodateien mit den Endungen .mp4, .avi oder .mov unterstützt werden.
Beispiele:
| Dateiname	            | Bedeutung                                 |
|-----------------------|-------------------------------------------|
| 001-Male-yawning.mp4	| Video mit ID 001, männlich, Gähnen        |
| 001-Male-normal.mp4	| Video mit ID 001, männlich, kein Gähnen   |
| 50-own-yawning.avi	| Eigenes Video, Gähnen                     |
| 51-own-normal.avi	    | Eigenes Video, kein Gähnen                |

Das Label wird automatisch aus dem Feld activity abgeleitet. Enthält activity den Begriff yawning, wird das Label 1 vergeben. Andernfalls wird das Label 0 vergeben.

### 4.2 Datenstruktur
Die aktuelle Datenstruktur lautet: \
data/ \
 videos/ \
  ...
 splits.csv \
src/ \
 data.py \
 evaluation.py \
 training.py \
 utils.py    \
main.py \
Tester.py \
best_model_final.pt \
checkpoints/ \
runs/ \
README.md \
(summarize_trials.py) \
(final_train_only.py) 

### 4.3 Daten-Split
Alle Videos liegen in data/videos, und die Aufteilung wird über die Datei data/splits.csv gesteuert.
Die Split-Datei enthält unter anderem folgende Spalten:
| Spalte	    | Bedeutung                                 |
|---------------|-------------------------------------------|
| id	        | Proband-ID                                |
| info_labels	| Zusatzinformationen aus dem Dateinamen    |
| activity	    | Aktivität aus dem Dateinamen              |
| filepath	    | Dateipfad zum Video                       |
| yawning	    | Binäres Label                             |
| filename	    | Dateiname                                 |
| split	        | Zugeordneter Split                        |

Die möglichen Werte für split sind:
| Split | Bedeutung             |
|-------|-----------------------|
| train	| Trainingsdaten        |
| val	| Validierungsdaten     |
| test	| unabhängige Testdaten |

Zusätzlich unterstützt der Code intern den Split trainval. Dieser kombiniert train und val und wird für Cross-Validation sowie für das finale Training verwendet. Der ursprüngliche Testsplit bleibt während dieser Schritte unverändert und wird ausschließlich für die finale Evaluation verwendet.
Die Aufteilung wurde gruppiert nach ID erzeugt. Dadurch wird verhindert, dass Videos derselben ID gleichzeitig in Trainings-, Validierungs- und Testdaten vorkommen, wodurch das Modell personenbezogene Merkmale lernen könnte.

### 4.4 Split-Statistik
Die vollständige Split-Statistik kann mit folgendem Befehl ausgegeben werden:
```
python -c "from src.data import load_split_csv, print_split_statistics; df=load_split_csv(); print_split_statistics(df)"
```
Der finale Split hat die folgende Aufteilung:
| Split	| Videos | Yawning | Non-Yawning | IDs |
|-------|--------|---------|-------------|-----|
| Train	| 206	 | 76	   | 130	     | 30  |
| Val	| 73	 | 27	   | 46	         | 11  |
| Test	| 65	 | 24	   | 41	         | 11  |

## 5. Datenvorverarbeitung
### 5.1 Frame-Extraktion
Aus jedem Video wird eine feste Anzahl T gleichmäßig verteilter Frames extrahiert. Die Anzahl der Frames wird über den Parameter
```
--num_frames 
```
gesteuert (Beispiel: --num_frames 32) \
Die Frames werden mit torchcodec aus den Videodateien geladen.

### 5.2 Vorverarbeitung für Training
Für Trainingsdaten werden folgende Schritte angewendet: 

1.	Resize auf 256 × 341. 
2.	Zufälliger horizontaler Flip. 
3.	Zufällige Rotation im Bereich von ungefähr ±10 Grad. 
4.	Color-Jitter für Helligkeit, Kontrast, Sättigung und Farbton. 
5.	Center-Crop auf 224 × 224. 
6.	Konvertierung in Tensoren. 
7.	Normalisierung mit ImageNet-Mittelwerten und Standardabweichungen. 

Die zufälligen Augmentationen werden identisch auf alle Frames eines Clips angewendet. Dadurch entstehen keine künstlichen Sprünge zwischen aufeinanderfolgenden Frames.

### 5.3 Vorverarbeitung für Validierung und Test
Für Validierungs- und Testdaten werden keine zufälligen Augmentationen verwendet. Die Schritte lauten: 

1.	Resize auf 256 × 341. 
2.	Center-Crop auf 224 × 224. 
3.	Konvertierung in Tensoren. 
4.	Normalisierung mit ImageNet-Mittelwerten und Standardabweichungen. 

Die verwendeten Normalisierungswerte sind:
| Kanal | Mittelwert | Standardabweichung |
|-------|------------|--------------------|
| R	    | 0.485	     | 0.229              |
| G	    | 0.456	     | 0.224              |
| B	    | 0.406	     | 0.225              |

## 6. Implementierung
6.1 Code-Struktur
Die wichtigsten Dateien des Projekts sind:
| Datei             | Aufgabe                                                                                   |
|-------------------|-------------------------------------------------------------------------------------------|
| src/data.py       | Laden der Videos, Erzeugen und Verwenden der Split-Datei, Preprocessing und Augmentation  |
| src/training.py	| Modellarchitektur, Training, Loss, Early Stopping, Scheduler und Gradient Clipping        |
| src/evaluation.py	| Berechnung von Metriken und Confusion Matrix                                              |
| src/utils.py      | Reproduzierbarkeit durch feste Seeds                                                      |
| main.py           | Hyperparameteroptimierung, Cross-Validation, finales Training und finaler Test            |
| Tester.py         | Separates Laden und Testen des gespeicherten finalen Modells                              |

### 6.2 Installation
Es wird empfohlen, eine virtuelle Python-Umgebung zu verwenden. \
Unter Windows PowerShell:
```
python -m venv .venv 
.venv\Scripts\activate
```
Unter Linux oder macOS:
```
python -m venv .venv 
source .venv/bin/activate
```

Die wichtigsten Abhängigkeiten sind:
| Bibliothek    | Verwendung                        |
|---------------|-----------------------------------|
| torch	        | Deep-Learning-Framework           |
| torchvision	| ResNet18 und Bildtransformationen |
| torchcodec	| Laden von Videodateien            |
| pandas	    | Verarbeitung der Split-Datei      |
| numpy	        | Numerische Operationen            |
| scikit-learn	| Cross-Validation und Metriken     |
| tqdm	        | Fortschrittsanzeige               |
| optuna	    | Hyperparameteroptimierung         |
| tensorboard	| Logging und Visualisierung        |
| torchinfo	    | optionale Modellzusammenfassung   |

Die final verwendeten Python-Abhängigkeiten sind in `requirements.txt` gespeichert und können mit folgendem Befehl installiert werden:
```
pip install -r requirements.txt 
```
Eine manuelle Installation ist ebenfalls möglich:
```
pip install torch torchvision torchcodec pandas numpy scikit-learn tqdm optuna tensorboard torchinfo
```
Je nach CUDA-Version sollte PyTorch gemäß der offiziellen PyTorch-Anleitung installiert werden: \
https://pytorch.org/get-started/locally/

### 6.3 Datensatz vorbereiten
Alle Videos müssen in folgendem Ordner liegen: \
data/videos/ \
Die Split-Datei kann mit folgendem Befehl erzeugt werden:
```
python -c "from src.data import create_split_csv; create_split_csv(force=True)"
```
Wenn die Split-Datei final erzeugt wurde, sollte sie nicht erneut mit force=True überschrieben werden. Dadurch bleibt der Testsplit über alle Experimente hinweg konstant.

### 6.4 Dataset-Test
Zum Testen, ob das Dataset korrekt geladen wird, kann folgender Befehl verwendet werden:
```
python -c "from src.data import YawDDDataset; ds=YawDDDataset('train', num_frames=8, train=False); x,y=ds[0]; print(x.shape, y)"
```
Eine erwartete Ausgabe ist:
```
torch.Size([8, 3, 224, 224]) tensor(...)
```
Damit ist geprüft, dass ein Video geladen, vorverarbeitet und als Tensor zurückgegeben wird.

## 7. Training
### 7.1 Trainingsaufruf
Ein kurzer technischer Testlauf kann mit folgendem Befehl gestartet werden:
```
python main.py --num_frames 8 --epochs 1 --n_trials 1 --n_splits_cv 3
```
Ein vollständigerer Trainingslauf kann beispielsweise so gestartet werden:
```
python main.py --num_frames 32 --epochs 20 --n_trials 10 --n_splits_cv 5
```
Eine gründlichere Hyperparametersuche kann beispielsweise so aussehen:
```
python main.py --num_frames 32 --epochs 25 --n_trials 20 --n_splits_cv 5
```

### 7.2 Hyperparameter
Folgende Hyperparameter werden mit Optuna optimiert:
| Hyperparameter    | Wertebereich                          |
|-------------------|---------------------------------------|
| batch_size        | 4 oder 8                              |
| freeze_backbone   | 0 oder 1                              |   
| lr                | logarithmisch zwischen 1e-5 und 2e-4  |
| dropout           | 0.2 bis 0.6                           |
| threshold         | 0.25 bis 0.35                         |

Zusätzlich werden folgende Kommandozeilenargumente verwendet:
| Argument      | Bedeutung	                    | Beispiel |
|---------------|-------------------------------|----------|
| --num_frames  | Anzahl Frames pro Video	    | 32       |
| --epochs      | maximale Anzahl Epochen	    | 20       |
| --n_trials    | Anzahl Optuna-Trials	        | 10       |
| --n_splits_cv	| Anzahl Cross-Validation-Folds	| 5        |
| --patience	| Early-Stopping-Patience	    | 10       |

### 7.3 K-Fold Cross-Validation
Die Hyperparameteroptimierung verwendet Cross-Validation auf dem kombinierten Split trainval.
Dabei wird bevorzugt StratifiedGroupKFold verwendet. Diese Methode versucht, die Klassenverteilung in den Folds ähnlich zu halten und zusätzlich dieselbe ID nicht gleichzeitig in Trainings- und Validierungsdaten eines Folds zu verwenden.
Falls StratifiedGroupKFold nicht verfügbar ist oder fehlschlägt, wird auf GroupKFold zurückgegriffen. Dadurch bleibt zumindest die Gruppierung nach ID erhalten.
Der unabhängige Testsplit wird während der Hyperparameteroptimierung nicht verwendet.

### 7.4 Early Stopping
Während der Cross-Validation wird Early Stopping anhand des Validation-F1-Scores verwendet. Wenn sich der Validation-F1 über mehrere Epochen nicht verbessert, wird das Training des aktuellen Folds beendet.
Das beste Modell eines Folds wird gespeichert unter: \
checkpoints/trial_X_fold_Y.pt \ 
Beim finalen Training auf trainval wird kein Early Stopping verwendet, weil kein separater Validierungssplit mehr vorhanden ist. Dadurch wird vermieden, dass Early Stopping auf Trainingsdaten durchgeführt wird.

### 7.5 Finales Training
Nach der Optuna-Hyperparameteroptimierung wird das finale Modell mit den besten Hyperparametern auf dem gesamten Split trainval trainiert.
Das finale Modell wird gespeichert als: \ 
best_model_final.pt \ 
Dieser Checkpoint enthält:
| Inhalt            | Beschreibung                                      |
|-------------------|---------------------------------------------------|
| model_state       | Modellgewichte                                    |
| dropout	        | verwendeter Dropout-Wert                          | 
| threshold	        | verwendeter Klassifikationsschwellwert            |
| freeze_backbone   | Information, ob der Backbone eingefroren wurde    |
| batch_size        | verwendete Batchgröße                             |
| lr                | verwendete Lernrate                               |
| num_frames        | Anzahl der Frames pro Video                       |
| cv_best_f1        | bester Cross-Validation-F1                        |
| final_train_f1    | finaler Trainings-F1                              |

## 8. Evaluation
### 8.1 Evaluationsablauf
Nach dem finalen Training wird das Modell auf dem unabhängigen Testsplit evaluiert. Dieser Testsplit wurde während der Hyperparameteroptimierung und während des finalen Trainings nicht verwendet.
Alternativ kann die Evaluation separat mit dem Tester gestartet werden:
```
python Tester.py
```
Der Tester lädt automatisch: \
best_model_final.pt \
und verwendet die im Checkpoint gespeicherten Werte für: 
1.	Dropout 
2.	Threshold 
3.	Anzahl Frames 
4.	Batchgröße 

### 8.2 Metriken
Für die Evaluation werden folgende Metriken verwendet:
| Metrik            | Bedeutung                                                                         |
|-------------------|-----------------------------------------------------------------------------------|
| Accuracy	        | Anteil korrekt klassifizierter Beispiele                                          |
| Precision	        | Anteil korrekt positiver Vorhersagen unter allen positiven Vorhersagen            |
| Recall	        | Anteil erkannter positiver Beispiele unter allen tatsächlich positiven Beispielen |
| F1-Score	        | Harmonisches Mittel aus Precision und Recall                                      |
| ROC-AUC	        | Trennfähigkeit über verschiedene Schwellenwerte                                   |
| PR-AUC	        | Fläche unter der Precision-Recall-Kurve                                           |
| Confusion Matrix	| Übersicht über TP, TN, FP und FN                                                  |

### 8.3 Definition der Metriken
Accuracy = (TP + TN) / (TP + TN + FP + FN) \
Precision = TP / (TP + FP) \
Recall = TP / (TP + FN) \
F1-Score = 2 · (Precision · Recall) / (Precision + Recall) \
Confusion Matrix:
| Ground Truth              | Vorhersage non-yawning    | Vorhersage yawning    |
|---------------------------|--------------------------:|----------------------:|
| Tatsächlich non-yawning	| TN                        | FP                    |
| Tatsächlich yawning       | FN                        | TP                    |

In der Konsolenausgabe wird die Confusion Matrix in folgender Form dargestellt: \
    [[TN FP] \
    [FN TP]]

## 9. Logging und Visualisierung
### 9.1 TensorBoard
Für Logging und Visualisierung wird TensorBoard verwendet. Während des Trainings werden unter anderem folgende Werte geloggt:

| TensorBoard-Tag   | Bedeutung                                     |
|-------------------|-----------------------------------------------|
| Loss/train        | Trainingsloss                                 |
| train/F1          | F1-Score auf Trainingsdaten ohne Augmentation |
| val/F1            | F1-Score auf Validierungsdaten                |
| train/Accuracy    | Accuracy auf Trainingsdaten                   |
| val/Accuracy      | Accuracy auf Validierungsdaten                |
| LearningRate      | aktuelle Lernrate                             |
| test/F1           | F1-Score auf Testdaten                        |

TensorBoard kann mit folgendem Befehl gestartet werden:
```
tensorboard --logdir runs
```
Anschließend kann die Oberfläche im Browser geöffnet werden: \
http://localhost:6006

## 10. Ergebnisse
Insgesamt wurden zwei größere Trainingsläufe durchgeführt. Der erste Lauf erfolgte auf einem CPU-basierten System und diente als erster vollständiger Trainings- und Evaluationsdurchlauf. Anschließend wurde ein zweiter, umfangreicherer Trainingslauf auf einem leistungsfähigeren Rechner mit NVIDIA RTX 4070 Laptop GPU durchgeführt.

Der zweite Lauf erzielte die besten Ergebnisse und wird daher als finales Modell betrachtet. Der erste Lauf wird im Folgenden als Vergleichslauf aufgeführt.


### 10.1 Vergleich der Trainingsläufe
| Merkmal                   | Vergleichslauf                                    | Finaler Lauf                  |
|---------------------------|--------------------------------------------------:|------------------------------:|
| Prozessor                 | Intel Core Ultra 9 285H                           | Intel Core i7-13650HX         |
| GPU                       | keine GPU verwendet                               | NVIDIA RTX 4070 Laptop, 8 GB  |
| RAM                       | 32 GB, 5600 MT/s                                  | 32 GB, 4800 MT/s              |
| Betriebssystem            | Windows 11                                        | Windows 11                    |
| Bester CV-F1              | 0.9349                                            | 0.9482                        |
| Finaler Test-F1           | 0.889                                             | 1.000                         |
| Finaler Test-Accuracy     | 0.908                                             | 1.000                         |
| Finaler Test-Recall       | 1.000                                             | 1.000                         |
| Finaler Test-Precision    | 0.800                                             | 1.000                         |
| Laufzeit                  | ca. 8 h 54 min für Hyperparameteroptimierung/CV   | 38 h 42 min 29 s gesamt       |

Obwohl der finale Lauf auf einer deutlich leistungsfähigeren GPU durchgeführt wurde, war die Gesamtlaufzeit höher. Dies liegt daran, dass der Lauf umfangreicher war und das finale Modell mit `freeze_backbone = 0` trainiert wurde. Dadurch wurden nicht nur der Klassifikationskopf und der letzte ResNet-Block, sondern der gesamte ResNet18-Backbone trainiert. Dies erhöht den Rechenaufwand deutlich. Da der Code ursprünglich für ein CPU-basiertes System mit `num_workers=0` ausgelegt war, wurde das Laden der Daten nicht parallelisiert. Dadurch konnte die GPU im finalen Lauf vermutlich nicht vollständig ausgelastet werden. Die GPU ermöglichte dennoch das Training eines umfangreicheren Modells mit vollständig trainierbarem Backbone und führte zu besseren Ergebnissen auf dem Testsplit.

Die Laufzeit hängt insbesondere von folgenden Faktoren ab: 

1.	Anzahl der Frames pro Video. 
2.	Anzahl der Epochen. 
3.	Anzahl der Optuna-Trials. 
4.	Anzahl der Cross-Validation-Folds. 
5.	Batchgröße. 
6.	Verwendete Hardware. 

Der finale Lauf lief über 10 Trials mit maximal 30 Epochen und 5 Folds, während der Vergleichslauf nur 5 Trials mit maximal 20 Epochen und 3 Folds verwendete.

Der zweite Trainingslauf erzielte sowohl in der Cross-Validation als auch auf dem finalen Testsplit bessere Ergebnisse. Besonders auffällig ist, dass das finale Modell im zweiten Lauf alle Testvideos korrekt klassifizierte.

### 10.2 Finale Hyperparameter
Die Hyperparameter wurden zunächst mithilfe von Optuna und gruppierter Cross-Validation auf dem kombinierten Split trainval bestimmt. Anschließend wurde mit dem besten gefundenen Parametersatz ein finales Modell auf dem gesamten trainval-Split trainiert. Der beste Cross-Validation-F1 des finalen Laufs betrug: \
Best trial CV-F1: 0.9482 \
Die besten Hyperparameter waren:
| Hyperparameter     | Wert                      |
|-------------------|---------------------------|
| batch_size        | 8                         |
| freeze_backbone   | 0                         |
| lr                | 0.00007868184508738092    |
| dropout           | 0.2                       |
| threshold         | 0.32797862391552335       |
| num_frames        | 32                        |


Da der Vergleichslauf vor dem finalen Speichern des Modells unterbrochen wurde, wurden die Hilfsskripte summarize_trials.py und final_train_only.py entwickelt und verwendet.
Der beste rekonstruierte Cross-Validation-F1 betrug: 0.9349

| Hyperparameter      | Wert          |
|---------------------|---------------|
| batch_size          | 8             |
| freeze_backbone     | 1             |
| lr                  | 0.00012836    |
| dropout             | 0.6           |
| threshold           | 0.31585       |
| num_frames          | 32            |

Im Gegensatz zum vorherigen Vergleichslauf wurde beim finalen Modell der Backbone nicht eingefroren. Dadurch konnten alle Parameter des ResNet18-Backbones während des Trainings angepasst werden.

### 10.2 Finales Training
Die Klassenverteilung im finalen Trainingssplit trainval war in beiden Läufen:
| Klasse        | Anzahl    |
|---------------|-----------|
| yawning	    | 103       |
| non-yawning   | 176       |
| Gesamt        | 279       |

Daraus ergab sich für die Loss-Funktion folgende Positiv-Gewichtung: \
pos_weight = 1.709

Diese Gewichtung wurde in BCEWithLogitsLoss verwendet, um die im Vergleich seltenere positive Klasse yawning stärker zu berücksichtigen.
Während des finalen Trainings wurde kein Early Stopping verwendet, da das Modell auf dem gesamten trainval-Split trainiert wurde und somit kein separater Validierungssplit für die Modellauswahl mehr zur Verfügung stand. Der Trainingsloss und der Trainings-F1 entwickelten sich im finalen Training wie folgt:
![Verlauf des Train Loss](/pictures/Loss_Final_Train.png) 
Abbildung 2: Verlauf des Train Loss während des finalen Trainings

![Verlauf des Train F1](/pictures/F1_Final_Train.png) 
Abbildung 3: Verlauf des F1-Scores während des finalen Trainings

Die Abbildungen zeigen, dass der Trainingsloss im Verlauf sinkt und die Trainingsmetriken ansteigen. Dies bestätigt, dass der Trainingscode das Modell erfolgreich optimiert.


Das finale Modell wurde erfolgreich gespeichert unter: \
best_model_final.pt

### 10.3 Finale Testergebnisse
Nach dem finalen Training wurde das Modell auf dem unabhängigen Testsplit evaluiert. Der Testsplit wurde weder während der Hyperparameteroptimierung noch während des finalen Trainings zur Modellauswahl verwendet.
Die Klassenverteilung im Testsplit war:
| Klasse        | Anzahl    |
|---------------|-----------|
| non-yawning	| 41        |
| yawning       | 24        |
| Gesamt	    | 65        |

Die finale Evaluation ergab folgende Metriken:
| Metrik	    | Vergleichslauf    | Finaler Lauf  |
|---------------|------------------:|--------------:|
| Accuracy	    | 0.908             | 1.000         |          
| Precision	    | 0.800             | 1.000         |
| Recall	    | 1.000             | 1.000         |
| F1-Score	    | 0.889             | 1.000         |
| ROC-AUC	    | 0.994             | 1.000         |
| PR-AUC	    | 0.991             | 1.000         |

Der finale Lauf erreichte auf dem unabhängigen Testsplit perfekte Werte für alle berechneten Metriken.

Die Confusion Matrizen der beiden Läufe ergaben:


![Confusion Matrix finaler Lauf](/pictures/Confusion_Matrix_final.png) 
Abbildung 4: Confusion Matrix des finalen Laufs auf dem unabhängigen Testsplit.

Das finale Modell klassifizierte alle 65 Testvideos korrekt. Es traten weder False Positives noch False Negatives auf.

![Confusion Matrix Vergleichslauf](/pictures/Confusion_Matrix_Vergleichslauf.png) 
Abbildung 5: Confusion Matrix des Vergleichslauf auf dem unabhängigen Testsplit.

Die Confusion Matrix des Vergleichslaufs zeigt, dass keine positive Testsequenz übersehen wurde. Gleichzeitig zeigen die sechs False Positives, dass das Modell teilweise auch andere Mundbewegungen oder gesichtsbezogene Veränderungen als Gähnen interpretiert. Der F1-Score von 0.889 zeigt insgesamt trotzdem eine gute Balance zwischen Precision und Recall.

Die in beiden Läufen sehr hohen Werte für ROC-AUC und PR-AUC deuten darauf hin, dass das Modell die beiden Klassen über die vorhergesagten Wahrscheinlichkeiten sehr gut trennt.


## 10.4 Interpretation der Ergebnisse
Das finale Modell erreichte auf dem Testsplit eine Accuracy, Precision, Recall und einen F1-Score von jeweils 1.000. Damit wurden alle yawning- und non-yawning-Videos im Testsplit korrekt klassifiziert.
Besonders relevant ist der Recall von 1.000, da im Kontext der Müdigkeitserkennung übersehene Müdigkeitsanzeichen potenziell sicherheitskritisch sein können. Im finalen Testlauf wurde kein tatsächliches Gähnen übersehen.
Gleichzeitig müssen die perfekten Testergebnisse vorsichtig interpretiert werden. Der Testsplit umfasst nur 65 Videos, davon 24 mit Gähnen. Bei einer vergleichsweise kleinen Testmenge können einzelne Eigenschaften des Datensatzes einen starken Einfluss auf die Metriken haben. Daher ist nicht auszuschließen, dass das Modell stark von den spezifischen Aufnahmebedingungen, Personen, Kameraperspektiven oder Bewegungsmustern des Datensatzes profitiert.
Eine weitere Validierung auf zusätzlichen, vollständig unabhängigen Videos wäre notwendig, um die Generalisierungsfähigkeit des Modells belastbarer zu bewerten.


## 11. Diskussion
### 11.1 Bewertung des Ansatzes
Das Modell verwendet einen zweistufigen Ansatz aus räumlicher Feature-Extraktion und temporaler Gewichtung. Der ResNet18-Backbone extrahiert visuelle Merkmale aus den einzelnen Frames. Der Attention-Mechanismus gewichtet anschließend die zeitliche Relevanz dieser Frames.

Ein Vorteil dieses Ansatzes ist, dass kein vollständig neues Videomodell von Grund auf trainiert werden muss. Durch den vortrainierten ResNet18-Backbone können bereits gelernte Bildmerkmale genutzt werden. Dies ist insbesondere bei einem kleinen Datensatz vorteilhaft.

Der finale Trainingslauf zeigte, dass das vollständige Fine-Tuning des Backbones eine deutliche Verbesserung gegenüber dem vorherigen Lauf mit teilweise eingefrorenem Backbone erzielen konnte. Während der Vergleichslauf einen Test-F1 von 0.889 erreichte, erzielte das finale Modell einen Test-F1 von 1.000.

Die Aussagekraft der finalen Testmetriken ist durch die begrenzte Größe des Testsets eingeschränkt. Der Testsplit umfasst 65 Videos, davon 24 mit Gähnen. Einzelne Fehlklassifikationen haben daher einen vergleichsweise starken Einfluss auf Precision, Accuracy und F1-Score.

### 11.2 Vergleich zwischen eingefrorenem und trainierbarem Backbone
Im Vergleichslauf wurde `freeze_backbone = 1` verwendet. Dabei wurde der ResNet18-Backbone weitgehend eingefroren, während der letzte ResNet-Block `layer4` weiterhin trainierbar blieb. Dieses Vorgehen reduziert die Anzahl der trainierbaren Parameter und kann bei kleinen Datensätzen Overfitting entgegenwirken.

Im finalen Lauf wurde hingegen `freeze_backbone = 0` gewählt. Dadurch wurde der gesamte ResNet18-Backbone trainiert. Dies ermöglichte dem Modell, die vortrainierten Bildmerkmale stärker an die spezifische Aufgabe der Gähn-Erkennung anzupassen.

Die Ergebnisse deuten darauf hin, dass das vollständige Fine-Tuning des Backbones für diesen Datensatz vorteilhaft war. Gleichzeitig steigt dadurch die Gefahr, dass das Modell sehr stark an die spezifischen Eigenschaften des Trainingsdatensatzes angepasst wird.


### 11.3 Diskussion der perfekten Testergebnisse
Das finale Modell erreichte auf dem Testsplit perfekte Ergebnisse (1.000). Eine perfekte Klassifikation ist grundsätzlich positiv, muss jedoch kritisch betrachtet werden. Der Testsplit umfasst nur 65 Videos. Dadurch kann bereits eine kleine Anzahl zusätzlicher oder schwieriger Testvideos die Metriken deutlich verändern.

Da das finale Modell den gesamten Backbone trainiert hat und der Trainingslauf umfangreicher war, besteht die Möglichkeit, dass das Modell sehr gut an die Verteilung des vorhandenen Datensatzes angepasst wurde. Obwohl die Aufteilung gruppiert nach ID erfolgte und somit keine identischen Personen-IDs zwischen Training, Validierung und Test auftreten, können dennoch ähnliche Aufnahmebedingungen, Kameraperspektiven, Hintergründe, Beleuchtung oder Bewegungsmuster über die Splits hinweg vorhanden sein.

Daher kann nicht ausgeschlossen werden, dass das Modell neben tatsächlichen Gähnmerkmalen auch datensatzspezifische Muster nutzt. Dies wird nicht als direkter Datenleckage-Fehler bewertet, zeigt aber die Notwendigkeit einer zusätzlichen externen Validierung.

### 11.4 Auswendiglernen und Overfitting
Der Begriff „Auswendiglernen“ beschreibt im Kontext neuronaler Netze, dass ein Modell nicht nur allgemeine Merkmale einer Klasse lernt, sondern sich stark an konkrete Eigenschaften der Trainingsdaten anpasst. Bei Videodaten können solche Eigenschaften beispielsweise sein:

- konstante Kamerapositionen,
- ähnliche Innenräume oder Hintergründe,
- ähnliche Beleuchtung,
- wiederkehrende Personenmerkmale,
- ähnliche Kopfhaltungen,
- ähnliche Videolängen oder Aufnahmebedingungen.

Durch die gruppierte Aufteilung nach ID wurde reduziert, dass Videos derselben Person gleichzeitig in Training und Test vorkommen. Dies verringert das Risiko direkter personenbezogener Datenleckage. Dennoch kann ein Modell bei einem kleinen und relativ homogenen Datensatz Muster lernen, die nicht vollständig allgemein auf neue Fahrsituationen übertragbar sind.

Die perfekten Testergebnisse des finalen Modells sollten daher als sehr gutes Ergebnis auf dem vorhandenen Testsplit interpretiert werden, nicht jedoch automatisch als Nachweis einer perfekten Generalisierung auf reale Fahrsituationen.

### 11.5 Herausforderungen
Eine Herausforderung besteht darin, dass Gähnen nur in bestimmten Abschnitten der Videos vorkommt. Zudem können andere Mundbewegungen, Sprechen oder Gesichtsausdrücke ähnlich interpretiert werden. Auch Beleuchtung, Kameraperspektive, Kopfbewegungen, Brillen, Bart oder individuelle Unterschiede zwischen Personen können die Erkennung erschweren.
Ein weiterer wichtiger Punkt ist die Datenaufteilung. Wenn Videos derselben Person gleichzeitig in Training und Test vorkommen, kann das Modell personenbezogene Merkmale lernen. Deshalb wurde eine gruppierte Aufteilung nach ID verwendet.
Des Weiteren wurden alle verwendeten Videos aus Sicherheitsgründen in stillstehenden Fahrzeugen aufgenommen. Eine realistische Darstellung alltäglicher Fahrsituationen ist daher nicht gegeben.

### 11.6 Grenzen des Modells
Die wichtigsten Grenzen des aktuellen Ansatzes sind:

1.	Der Datensatz ist vergleichsweise klein.
2.	Das Modell verwendet vollständige Frames und keine explizite Gesichtserkennung.
3.	Die Mundregion wird nicht separat lokalisiert.
4.	Die Augmentationen sind relativ einfach.
5.	Der Attention-Mechanismus betrachtet nur die zeitliche Relevanz, aber nicht direkt die räumlichen Bildbereiche.
6.	Das Modell betrachtet nur Gähnen und keine weiteren Müdigkeitsmerkmale wie Blinzeln, Blickrichtung oder Kopfnicken.
7.  Eine externe Validierung auf vollständig neuen Aufnahmebedingungen wurde bisher nicht durchgeführt.

### 11.7 Mögliche Erweiterungen
Mögliche Erweiterungen für zukünftige Arbeiten sind:

1.	Gesichtserkennung und Cropping auf die Gesichtsregion.
2.	Separater Fokus auf die Mundregion.
3.	Integration weiterer Müdigkeitsmerkmale.
4.	Multimodaler Ansatz mit Audiointegration
5.	Vergleich mit LSTM-, GRU- oder Transformer-Modellen.
6.	Vergleich mit 3D-CNNs.
7.	Visualisierung der Attention-Gewichte über die Zeit.
8.	Grad-CAM zur räumlichen Interpretierbarkeit.
9.	Systematische Threshold-Analyse.
10.	Erweiterung des Datensatzes um mehr eigene Videos.
11.	Evaluation auf vollständig neuen Probanden.
12. Test auf realistischeren Fahrsituationen mit variierenden Straßen-, Licht- und Kamerabedingungen.

## 12. Vergleich mit anderen Ansätzen
Ein direkter Vergleich mit anderen Modellen wurde bisher noch nicht durchgeführt. Mögliche Vergleichsmodelle wären: 

1.	ResNet18 mit Mittelwertbildung über alle Frame-Features. 
2.	ResNet18 mit Max-Pooling über die Zeit. 
3.	ResNet18-Features mit LSTM. 
4.	ResNet18-Features mit GRU. 
5.	3D-CNN. 
6.	Video Transformer. 
7.	Klassische Verfahren mit Mundregionserkennung und geometrischen Merkmalen.

Ein besonders naheliegender Vergleich wäre ein Modell mit einfacher Mittelwertbildung über alle Frame-Features. Dadurch könnte untersucht werden, ob der Attention-Mechanismus gegenüber einfachem Average Pooling einen messbaren Vorteil bringt.

## 13. Reproduzierbarkeit
Zur Verbesserung der Reproduzierbarkeit werden Seeds für Python, NumPy und PyTorch gesetzt. Die entsprechende Funktion befindet sich in: \
src/utils.py \
Der Seed wird in main.py gesetzt: \
setup_env(seed=0) \
Zusätzlich wird die Split-Datei gespeichert: \
data/splits.csv \
Diese Datei sollte für finale Experimente nicht mehr verändert werden.

## 14. Hinweise zur Konsolenausgabe
Beim Start von TensorBoard oder bei Verwendung des TensorBoard-Writers können TensorFlow-bezogene oneDNN-Hinweise erscheinen. Diese Meldungen stammen nicht aus dem eigentlichen PyTorch-Modelltraining und beeinflussen die Modelllogik nicht.
Um diese Meldungen zu unterdrücken, wurde vor dem TensorBoard-Import folgende Umgebungsvariable gesetzt:
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
Diese Einstellung unterdrückt TensorFlow-Info- und Warnmeldungen, behebt aber keine echten Programmfehler.

## 15. Hilfsskript summarize_trials.py
Das Skript summarize_trials.py dient dazu, abgeschlossene Optuna-Trials nachträglich aus den TensorBoard-Logs auszuwerten. Das Skript ist ein Analyse- und Wiederherstellungswerkzeug und wird nicht für einen regulären Trainingslauf benötigt.
Während der Hyperparameteroptimierung werden für jeden Trial und jeden Cross-Validation-Fold TensorBoard-Logs gespeichert. Falls die aggregierten Werte wie cv_mean_val_f1 nicht direkt in der TensorBoard-HParams-Tabelle sichtbar sind, können sie mit diesem Skript rekonstruiert werden.
Das Skript liest dazu aus den Ordnern: \
runs/optuna_trial_X/fold_Y/ \
die geloggten Werte des Tags: \
val/F1 \
aus. Für jeden Fold wird der beste erreichte Validation-F1 bestimmt. Anschließend werden der Mittelwert und die Standardabweichung über alle Folds eines Trials berechnet.
Dadurch lässt sich nachträglich bestimmen, welcher Optuna-Trial den besten mittleren Validation-F1 erreicht hat.
Beispielaufruf bei 5 Cross-Validation-Folds:
```
python summarize_trials.py --expected_folds 5
```
Beispielaufruf bei 3 Cross-Validation-Folds:
```
python summarize_trials.py --expected_folds 3
```
Die Ausgabe enthält unter anderem: \
Trial 0 \
Fold 0: best val/F1 = ... \
Fold 1: best val/F1 = ... \
Ergebnis: mean=..., std=..., folds=5/5, VOLLSTÄNDIG \
Am Ende wird der beste vollständige Trial ausgegeben: \
Bester vollständiger Trial \
Trial ID : ... \
cv_mean_val_f1 : ... \
cv_std_val_f1 : ... \
Dieses Skript ist besonders nützlich, wenn ein langer Trainingslauf abgebrochen wurde, bevor main.py das finale Modell speichern konnte, aber bereits TensorBoard-Logs der Cross-Validation vorhanden sind.


## 16. Hilfsskript final_train_only.py
Das Skript final_train_only.py dient dazu, das finale Training separat nachzuholen. Es wird verwendet, wenn die Hyperparameter bereits bekannt sind, aber das finale Modell best_model_final.pt noch nicht erzeugt wurde. 
Dies kann zum Beispiel passieren, wenn ein langer Lauf von main.py während oder nach der Cross-Validation abgebrochen wurde. In diesem Fall existieren häufig bereits Fold-Checkpoints und TensorBoard-Logs, aber noch kein finaler Checkpoint.
Das Skript trainiert ein neues Modell mit den angegebenen Hyperparametern auf dem gesamten Split: \
trainval\
Dabei werden train und val gemeinsam verwendet. Anschließend wird das resultierende Modell als finaler Checkpoint gespeichert: \
best_model_final.pt \
Zusätzlich wird das Modell direkt auf dem unabhängigen Testsplit evaluiert. \
Das Skript verwendet keine erneute Hyperparameteroptimierung, sondern trainiert ausschließlich mit manuell übergebenen Hyperparametern. \
Der in diesem Projekt für den Vergleichslauf verwendete Aufruf lautete:
```
python final_train_only.py --num_frames 32 --epochs 20 --batch_size 8 --freeze_backbone 1 --lr 0.00012836 --dropout 0.6 --threshold 0.31585 --cv_best_f1 0.9349
```
Das Skript erzeugt nach erfolgreichem Training folgende Datei: \
best_model_final.pt \
Dieser Checkpoint enthält neben den Modellgewichten auch die wichtigsten Hyperparameter.



## 17. Quellen

1.	YawDD: Yawning Detection Dataset. YawDD: Yawning Detection Dataset | IEEE DataPort
2.	He, K., Zhang, X., Ren, S., Sun, J.: Deep Residual Learning for Image Recognition. IEEE Conference on Computer Vision and Pattern Recognition, 2016.
3.	PyTorch Documentation: https://pytorch.org/docs/stable/index.html
4.	Torchvision Documentation: https://pytorch.org/vision/stable/index.html
5.	Optuna Documentation: https://optuna.readthedocs.io/
6.	TensorBoard Documentation: https://www.tensorflow.org/tensorboard
7.	scikit-learn Documentation: https://scikit-learn.org/stable/
8.  Confusion Matrix Generator: https://www.damianoperri.it/public/confusionMatrix/index.php


## 18. Anhang
![Accuracy finales Training](/pictures/Accuracy_Final_Train.png) 
Abbildung 6: Verlauf der Accuracy während des finalen Trainings.

![Precision finales Training](/pictures/Precision_Final_Train.png) 
Abbildung 7: Verlauf der Precision während des finalen Trainings.

![Recall finales Training](/pictures/Recall_Final_Train.png) 
Abbildung 8: Verlauf des Recalls während des finalen Trainings.