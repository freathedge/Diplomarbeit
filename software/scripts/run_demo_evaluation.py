"""Lädt (oder generiert) das Demo-Szenario, lässt alle Detektoren laufen
und druckt eine Precision/Recall/F1-Vergleichstabelle - Basis für die
Ergebnisse in Kapitel 5 der DA.

Aufruf:
    cd software
    python -m scripts.run_demo_evaluation
"""
from pathlib import Path

import pandas as pd

from sensorwarn.detectors import DETECTORS
from sensorwarn.metrics import evaluate_all
from scripts.generate_demo_dataset import build_demo_scenario

pd.set_option("display.width", 120)


def main():
    data, labels = build_demo_scenario()

    ground_truth = {}
    for sensor in data.columns:
        sensor_labels = labels[labels["sensor"] == sensor]
        mask = pd.Series(False, index=data.index)
        for _, row in sensor_labels.iterrows():
            mask.loc[row["start_time"] : row["end_time"]] = True
        ground_truth[sensor] = mask

    for sensor in data.columns:
        alarms = {name: fn(data[sensor]) for name, fn in DETECTORS.items() if name != "dropout" or True}
        result = evaluate_all(alarms, {sensor: ground_truth[sensor]})
        print(f"\n=== Sensor: {sensor} ===")
        print(result[["detector", "precision", "recall", "f1", "fpr", "tp", "fp", "fn"]].to_string(index=False))


if __name__ == "__main__":
    main()
