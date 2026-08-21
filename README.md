# PWADL_SoSe26 - Dokumentation:

## Kapitel 1 - Problembeschreibung:

### Ausgangslage:	

Laut statistischem Bundesamt gab es 2020 1448 Unfälle in Deutschland, die durch Müdigkeit am Steuer entstanden sind.
Die Dunkelziffer wird deutlich höher geschätzt, da unter anderem 26% der befragten Fahrer im Rahmen einer Befragung
des DVR zugaben schon einmal am Steuer eingeschlafen zu sein. Verschiedenster Studien zufolge sind bis zu 25% der
Todesopfer im Straßenverkehr auf Müdigkeitsunfälle zurückzuführen.
[https://www.dvr.de/ueber-uns/positionen-des-dvr/beschluesse/muedigkeit-im-strassenverkehr]
		
Seit Juli 2024 sind Müdigkeitswarner für neue Autos in der EU Pflicht. Um die Müdigkeitserkennung zu verbessern, wird
weltweit nach Möglichkeiten der besseren Erkennung gesucht. Der Einsatz von neuronalen Netzen gilt als eine der
vielversprechenden Varianten der Müdigkeitserkennung. Diese werden klassisch zur Videoanalyse ausgelegt und auf 
verschiedensten Datensätzen trainiert. Einer der größten Datensätze in diesem Bereich ist der YawDD-Datensatz eingereicht
von Shervin Shirmohammadi im Jahre 2020. Dieser wird bis heute (08/2026) aktuell gehalten.
[https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset]

Dieses Projekt befasst sich mit der Müdigkeitserkennung durch ein ResNet, um visuelle Anzeichen auf Müdigkeit in Videodateien
zu erkennen. Als Datensatz dient der YawDD-Datensatz, welcher mit eigenen Aufnahmen erweitert wurde.


### Modell-Architektur + Mathematische Beschreibung:
		
#### Feature-Extraktion:

Als Backbone wird ein vortrainiertes ResNet18-Modell verwendet. Die finale Klassifikationschicht wird entfernt,
sodass ausschließlich die Merkmalsextraktion genutzt wird.

Input:

        X ∈ R^{B * T * C * H * W}

mit

- B = Batchgröße
- T = Anzahl Frames
- C = Farbkanäle
- H & W = Bildgröße

Jedes Frame wird unabhängig durch ResNet18 verarbeitet.

Output:

        F ∈ R^{B * T * D}

mit

- D = 512

als Feature-Dimension.

#### Temporal Attention Pooling:

Da nicht alle Frames einer Videosequenz gleich relevant sind, wird ein Attetion-Mechanismus zu Gewichtung
der Frames verwendet.

Für jedes Frame t:

        s_t ​= W_2(tanh(W_1 *​f*_t​))
 
Anschließend werden die Attention-Gewichte berechnet:

        α_t​ = e^(s_t) / sum_j{e^(s_j)}
 
Die Videorepräsentation ergibt sich durch:

        z = sum_(t=1){​α_t ​f_t}

#### Klassifikationskopf

Die aggregierten Features werden durch mehrere Fully-Connected-​Layers verarbeitet:

        512 -> 256 -> 128 -> 1

mit:

- Batch Normalization
- ReLU
- Dropout

Der Finale Logit lautet:

        y = f(z)

Die Wahrscheinlichkeit eines Gähnens wird über die Sigmoid-Funktion berechnet:

        p(y=1) = σ(y)

#### Lossfunction

Verwendet wird:

        L = BCEWithLogitsLoss()

mit einer Klassengewichtung:

        pos_weight = 2.0

um die positive Klasse stärker zu gewichten.

## Kapitel 2 - Datensatz:

### Beschreibung des Datensatzes:

Für die Durchführung des Projekts wurde hauptsächlich der **YawDD Mirror Videos Datensatz** verwendet. Die Quelle des Datensatzes ist in Kapitel 1 angegeben. Zusätzlich wurden eigene Videoaufnahmen erstellt und in den Datensatz integriert, um die Anzahl der Trainingsbeispiele zu erhöhen und zusätzliche Variationen hinsichtlich Personen, Beleuchtung und Aufnahmebedingungen abzudecken.

Der Datensatz enthält Videos von Personen sowohl **mit Gähnen** als auch **ohne Gähnen**. Darüber hinaus wurden verschiedene Erscheinungsformen berücksichtigt, sodass sowohl Aufnahmen von Personen **ohne Brille**, **mit Brille** sowie **mit Sonnenbrille** enthalten sind. Dies erhöht die Robustheit des Modells gegenüber unterschiedlichen realen Einsatzbedingungen.

Alle Videos besitzen eine Länge von ungefähr **30 Sekunden** und liegen in einer Auflösung von **360p** vor. Die Videos wurden mit einer festen Kameraposition aufgenommen und zeigen das Gesicht der Person während verschiedener Aktivitäten.

Die Benennung der Videodateien folgt einem einheitlichen Schema:

        ID-Spezifikation-Label

### Daten-Split:

Eine zentrale Herausforderung bei biometrischen Datensätzen besteht darin, sogenannte Data Leaks zu verhindern. Werden Videos derselben Person gleichzeitig für Training und Test verwendet, kann das Modell personenbezogene Merkmale lernen und dadurch unrealistisch gute Ergebnisse erzielen.

Um dieses Problem zu vermeiden, wurde eine gruppenbasierte Aufteilung des Datensatzes implementiert. Die Gruppen werden durch die im Dateinamen enthaltenen Personen-IDs definiert. Für die Aufteilung wird die Klasse GroupShuffleSplit aus Scikit-Learn eingesetzt.

Die Daten werden in folgende Teilmengen aufgeteilt:

- Trainingsdaten: 70 %
- Validierungsdaten: 15 %
- Testdaten: 15 %

Als Group dient die in der Videobeschriftung enthaltene ID. Dadurch wird sichergestellt, dass eine Person ausschließlich in einem einzigen Datensatzsplit vorkommt. Zu beachten ist, dass die Nummerierung durch die ID bei Damen und Herren seperat beginnt, sodass die ID´s nicht einzigartig im Datensatz sind. Hier ist dies jedoch keine Einschränkung, sondern eine Möglichkeit die Gleichverteilung bezüglich des Geschlechtes beim Training zu garantieren.

### Vorverarbeitung:

Nicht Dashcamanteil von YawDD da Datenmenge sonst für Hardware zu groß, kann aber nach richtiger Videobeschriftung damit erweitert werden, Eigene Videos wurden von 1080p auf 360p heruntergebrochen

### Vorbereitung der Daten:

        1. Data-Ordner in Projekt-Ordner erstellen
        2. YawDD-Dataset herunterladen
        3. Videos aus Mirror-Ordner in Data-Ordner verschieben
        4. ggf. Eigene Videos in Data-Ordner
        5. ggf. DashCam Videos aus YawDD hinzufügen
        (bei 4. und 5. Vorverarbeitung beachten)

### Visualisierung:
        
        ???


## Kapitel 3 - Code und Anweisungen zum Ausführen des Repositorys:

### Abhängigkeiten:

        1. Python und VSC installieren
        2. Projekt aus Repository in Ordner lokal speichern und mit VSC aufrufen
        3. VENV erschaffen
        4. Bibliotheken über pip installieren
        5. FFMPEG einrichten und zum Path hinzufügen
        6. GitIgnore einrichten
        7. Debugger einrichten
        8. Testrun

### Trainingscode und Hyperparameter:

### Evaluationscode:


## Kapitel 4 - Ergebnisse und Diskussion:

Laufzeit und Ressourcen: FOLGT

Verwendete Metriken: Accuracy, F1Score, GGF WEITERE

Darstellung der Metriken: FOLGT

Vergleich: IN ZITARO NACHSCHAUEN

Besonderheiten und Schwächen: VIELE



----------------------------------------------------------------------------

Das Projekt ist modular aufgebaut und besteht aus den folgenden Modulen:
		
		Data.py:
                Dieses Modul importiert die im Data-Ordner zur Verfügung gestellt Daten, erkennt die Label und fasst Sie in einem
                Dataframe zusammen.Dann werden die Daten über einen Group-Shuffle-Split in Trainings-, Validierungs- und Testdaten
                unterteilt. Abschließend folgt die Datenbearbeitung.
	
		Evaluation.py:
                Bei der Evaluation findet die Bewertung des Modells durch Vergleich seiner Antworten mit den gespeicherten Labels statt.
                Dargestellt wird diese während des Trainings durch die Ausgabe von Accuracy und F1Score von Trainingsdaten und Validierungsdaten.

		Training.py:
                Beinhaltet die Struktur des neuronalen Netzes, die Spezifikationen des Trainers und den Trainingsloop. 

		Utils.py:
                Utils beinhaltet die für den Setup wichtigen Daten wie zum Beispiel den benutzen Seed.

		Config.py:
                TODO

		Main.py:
                Kern des neuronalen Netzes. Hier werden die anderen Module abgerufen und das neuronale Netz zum Leben erwacht.



                YawDD Mirror Videos, Quelle siehe oben, eigene Videos, Videos mit Gähnen und ohne Gähnen, Videos mit Brille, Sonnenbrille und ohne Brille, Videobeschriftung besteht aus ID/Spezifikation/Label, 30Sek Länge, 360p

