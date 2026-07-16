# Digitales Frühwarnsystem für Sensorfehler im Wasserkraftwerk

## Ziel

Automatische Erkennung von fehlerhaften oder eingefrorenen Sensoren (Temperatur, Druck, Durchfluss), um Wartungskosten zu senken. ABB hat sowas verbaut (ROWII).

### Kurzbeschreibung

* Laufende Überwachung der Sensordaten auf **Drift, Rauschen, konstante Werte oder Ausreißer**.
* Umsetzung mit **Moving Average**, **EWMA**, und **z-Score**-Erkennung.
* Anzeige fehlerhafter Sensoren in einem Dashboard mit Status-Ampeln.

---

## Technische Umsetzung

### Software & Tools (Vorschlag)

* Python mit `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `streamlit`.
* Versionsverwaltung mit GitHub.
* Jupyter Notebook für Analyse.

### Datenerzeugung (synthetisch)

```python
# Beispiel: simulierte Sensordaten
import numpy as np, pandas as pd

t = pd.date_range("2025-01-01", periods=5000, freq="T")
flow = 100 + np.sin(np.linspace(0,30,5000))*2 + np.random.normal(0,0.3,5000)
temp = 25 + 0.05*np.sin(np.linspace(0,10,5000)) + np.random.normal(0,0.2,5000)

# Anomalien einfügen
flow[3000:3020] -= 10
temp[4000:4020] += 5

data = pd.DataFrame({'flow':flow, 'temp':temp}, index=t)
data.to_csv("hydro_data.csv")
```

Anschließend wurden Anomalien eingefügt.

## Erklärung Daten

### Datenpunkte

* Flow - Durchfluss Sensor
* temp - temperatur Sensor
* rpm - Drehzahl
* vib - Vibration

### Anomalien

* spike_alarm - flow, press
* drift_alarm - flow, press
* frozen_alarm - press, vibration
* dropout flow, temp
* Bonus: correlated

---

## Aufbau der Diplomarbeit (Vorschlag - chatgpt)

1. **Einleitung**

   * Motivation (Bedeutung von Zustandserkennung in Wasserkraftwerken)
   * Zielsetzung und Aufgabenstellung

2. **Theoretische Grundlagen**

   * Funktionsweise eines Wasserkraftwerks
   * Sensorik und Messgrößen
   * Grundlagen der Anomalieerkennung (PCA, Thresholds, RMS etc.)

3. **Methodik und Konzept**

   * Datenmodell und Signalverarbeitung
   * Beschreibung der Algorithmen
   * Aufbau der Softwarearchitektur

4. **Implementierung**

   * Python-Code / Software-Design
   * Benutzeroberfläche (Dashboard oder Notebook)
   * Tests mit simulierten Daten

5. **Ergebnisse und Evaluation**

   * Darstellung von Diagrammen und Alarmfällen
   * Diskussion über Erkennungsrate und Fehlalarme

6. **Schlussfolgerung und Ausblick**

   * Bewertung der Methode
   * mögliche Erweiterungen (z. B. KI, Echtzeitbetrieb)

---

## Empfohlene Kombination für 2 Personen

**Thema:**

> „Entwicklung eines Systems zur Anomalieerkennung in Sensordaten eines Wasserkraftwerks mit PCA und Schwellenwertverfahren“

**Projektziele:**

* Sensordaten simulieren oder von einer Modellanlage erfassen
* PCA + Schwellenwertverfahren implementieren
* Anomalien automatisch erkennen und visualisieren
* Software-Prototyp (Streamlit-App oder Jupyter Notebook)

**Aufgabenteilung:**

* *Person A:* Datenerfassung, Analyse, Dokumentation Grundlagen
* *Person B:* Algorithmik, Software, Visualisierung, Dokumentation Implementierung

**Erwartetes Ergebnis:**
Funktionierendes Programm + klar strukturierte Diplomarbeit + Demonstration am Beispiel-Datensatz oder Modellanlage.

---

Anmerkung:

Alle Anomalien konnten mit `data_generation_and_detection.py` detektiert werden - die Analyse hat eine hohe Sensivität es sind also einige False - Positives dabei. Lieber zu viel Prüfen.

---
