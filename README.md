# PWADL_SoSe26

# Dokumentation:

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

Als Backbone wird ein vortrainiertes ResNet18-Modell verwendet. Die finale Klassifikationschicht wird entfernt, sodass
ausschließlich die Merkmalsextraktion genutzt wird.

Input:

**X** \in \mathbb{R}^{B \times T \times C \times H \times W}


mit

- B = Batchgröße
- T = Anzahl Frames
- C = Farbkanäle
- H & W = Bildgröße

Jedes Frame wird unabhängig durch ResNet18 verarbeitet.

Output:



## Kapitel 2 - Datensatz:

Beschreibung des Datensatzes:
YawDD Mirror Videos, Quelle siehe oben, eigene Videos, Videos mit Gähnen und ohne Gähnen, Videos mit Brille, Sonnenbrille und ohne Brille, Videobeschriftung besteht aus ID/Spezifikation/Label, 30Sek Länge, 360p

Daten-Split: 70/15/15 über GroupShuffleSplit mit Group:=ID

Vorverarbeitung: Nicht Dashcamanteil von YawDD da Datenmenge sonst für Hardware zu groß, kann aber nach richtiger Videobeschriftung damit erweitert werden, Eigene Videos wurden von 1080p auf 360p heruntergebrochen

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
