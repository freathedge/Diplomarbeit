# sensorwarn

Digitales Frühwarnsystem für Sensorfehler im Wasserkraftwerk (DA Buder/Frick,
Auftraggeber illwerke vkw AG). Dieser Ordner enthält den Softwareteil
(Adrian Buder): Datengenerierung, Fehlerinjektion, Detektionsalgorithmen,
Evaluation.

## Setup

```bash
cd software
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Struktur

```
software/
  src/sensorwarn/
    generator.py   Synthetische, gekoppelte Basissensordaten (flow, press, temp, rpm, vib)
    faults.py      Fehlerinjektion (spike, drift, frozen, dropout, correlated) + Ground-Truth-Labels
    detectors.py   MA, EWMA, z-Score, Frozen-, Dropout-, Drift- und PCA-Detektor
    metrics.py     Precision/Recall/F1/FPR gegen Ground-Truth-Labels
    dashboard_logic.py  Ampel-Status + Alarmtabelle (Streamlit-unabhängig, getestet)
  dashboard/
    app.py         Streamlit-UI: Zeitreihen, Status-Ampeln, Alarmtabelle, CSV-Export
  scripts/
    generate_demo_dataset.py    erzeugt data/demo_sensor_data.csv + demo_labels.csv
    run_demo_evaluation.py      führt alle Detektoren aus und druckt Vergleichstabelle
  tests/           pytest-Tests für Generator, Fehlerinjektor, Detektoren, Dashboard-Logik
  data/            generierte CSVs (nicht versioniert)
```

## Schnellstart

```bash
python scripts/generate_demo_dataset.py
python scripts/run_demo_evaluation.py
pytest
streamlit run dashboard/app.py
```

Das Dashboard läuft unter http://127.0.0.1:8501 (nicht `localhost`, siehe
`.streamlit/config.toml`). Nur aus `software/` starten, sonst wird die Config
nicht geladen.

## Offene Punkte (siehe Softwareplan_Buder.md)

- Parameterstudie Schwellenwert vs. Erkennungsrate
- YAML-Konfiguration statt hartcodierter Szenarien
- Alarm-Priorisierung / Entprellung vor der Ampel-Logik
