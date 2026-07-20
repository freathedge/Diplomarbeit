"""Evaluation der Detektoren gegen Ground-Truth-Labels (Kapitel 5 der DA:
Precision, Recall, F1, False-Positive-Rate, Detektionsverzögerung).
"""
from __future__ import annotations

import pandas as pd


def confusion_counts(alarm: pd.Series, ground_truth: pd.Series) -> dict[str, int]:
    """Sample-weise Vergleich von Alarm-Maske und Ground-Truth-Maske."""
    alarm = alarm.reindex(ground_truth.index).fillna(False)
    tp = int((alarm & ground_truth).sum())
    fp = int((alarm & ~ground_truth).sum())
    fn = int((~alarm & ground_truth).sum())
    tn = int((~alarm & ~ground_truth).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall_f1(alarm: pd.Series, ground_truth: pd.Series) -> dict[str, float]:
    c = confusion_counts(alarm, ground_truth)
    precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = c["fp"] / (c["fp"] + c["tn"]) if (c["fp"] + c["tn"]) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr, **c}


def detection_delay(alarm: pd.Series, labels: pd.DataFrame, sensor: str) -> list[float]:
    """Für jeden Fehler-Eintrag eines Sensors: wie viele Samples vergehen
    zwischen Fehlerbeginn und erstem Alarm (NaN, falls nie erkannt)."""
    delays: list[float] = []
    sensor_labels = labels[labels["sensor"] == sensor]
    for _, row in sensor_labels.iterrows():
        window = alarm.loc[row["start_time"] :]
        hit = window[window]
        if hit.empty:
            delays.append(float("nan"))
        else:
            delay = (hit.index[0] - row["start_time"]) / (alarm.index[1] - alarm.index[0])
            delays.append(float(delay))
    return delays


def evaluate_all(
    alarms: dict[str, pd.Series], ground_truth: dict[str, pd.Series]
) -> pd.DataFrame:
    """Vergleichsmatrix: Zeile je Detektor/Sensor-Kombination mit
    Precision/Recall/F1 - Basis für die Auswertungstabelle in der DA."""
    rows = []
    for name, alarm in alarms.items():
        for sensor, gt in ground_truth.items():
            m = precision_recall_f1(alarm, gt)
            rows.append({"detector": name, "sensor": sensor, **m})
    return pd.DataFrame(rows)
