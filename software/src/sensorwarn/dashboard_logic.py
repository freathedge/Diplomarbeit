"""Reine Logik für das Dashboard (keine Streamlit-Abhängigkeit), damit sie
mit pytest getestet werden kann. `dashboard/app.py` ist nur die dünne
Streamlit-Hülle darüber.
"""
from __future__ import annotations

import pandas as pd

from sensorwarn.detectors import (
    drift_detector,
    dropout_detector,
    frozen_detector,
    moving_average_detector,
    zscore_detector,
)

# Reihenfolge ist relevant für die Alarmtabelle (Anzeige-Priorität)
DETECTOR_PIPELINE = {
    "dropout": lambda s: dropout_detector(s),
    "frozen": lambda s: frozen_detector(s, window=10),
    "drift": lambda s: drift_detector(s, window=120, slope_threshold=0.02),
    "zscore": lambda s: zscore_detector(s, window=60, threshold=3.0),
    "moving_average": lambda s: moving_average_detector(s, window=30, threshold=3.0),
}


def compute_sensor_alarms(
    df: pd.DataFrame, sensor: str, params: dict | None = None
) -> dict[str, pd.Series]:
    """Alarm-Maske je Detektor für einen einzelnen Sensor.

    `params` erlaubt es, einzelne Detektor-Parameter zu überschreiben, z.B.
    ``{"zscore": {"window": 40, "threshold": 2.5}}`` – wird vom Dashboard
    genutzt, um die Regler mit den Detektoren zu verbinden.
    """
    params = params or {}
    series = df[sensor]
    result = {}
    result["dropout"] = dropout_detector(series)
    result["frozen"] = frozen_detector(series, **{"window": 10, **params.get("frozen", {})})
    result["drift"] = drift_detector(series, **{"window": 120, "slope_threshold": 0.02, **params.get("drift", {})})
    result["zscore"] = zscore_detector(series, **{"window": 60, "threshold": 3.0, **params.get("zscore", {})})
    result["moving_average"] = moving_average_detector(
        series, **{"window": 30, "threshold": 3.0, **params.get("moving_average", {})}
    )
    return result


def combined_alarm(alarms: dict[str, pd.Series]) -> pd.Series:
    """OR-Verknüpfung aller Detektor-Alarme eines Sensors."""
    combined = None
    for mask in alarms.values():
        combined = mask.copy() if combined is None else (combined | mask)
    return combined


def status_light(combined: pd.Series, recent_window: int = 30, lookback_window: int = 300) -> str:
    """Ampel-Status aus dem letzten Ausschnitt der Alarm-Maske.

    - rot:  Alarm in den letzten `recent_window` Samples
    - gelb: kein akuter Alarm, aber innerhalb `lookback_window` mind. einer
    - grün: keine Alarme im Beobachtungsfenster
    """
    if combined.empty:
        return "green"
    recent = combined.iloc[-recent_window:]
    lookback = combined.iloc[-lookback_window:]
    if recent.any():
        return "red"
    if lookback.any():
        return "yellow"
    return "green"


STATUS_EMOJI = {"green": "🟢", "yellow": "🟡", "red": "🔴"}


def _runs(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Wandelt eine boolesche Maske in zusammenhängende (start, end)-Intervalle um."""
    if not mask.any():
        return []
    shifted = mask.ne(mask.shift()).cumsum()
    runs = []
    for _, group in mask.groupby(shifted):
        if group.iloc[0]:
            runs.append((group.index[0], group.index[-1]))
    return runs


def alarm_table(all_sensor_alarms: dict[str, dict[str, pd.Series]]) -> pd.DataFrame:
    """Baut eine flache Tabelle (sensor, detector, start, end, duration_samples)
    aus den Alarm-Masken aller Sensoren/Detektoren - Basis für Anzeige und Export."""
    rows = []
    for sensor, detector_alarms in all_sensor_alarms.items():
        for detector_name, mask in detector_alarms.items():
            for start, end in _runs(mask):
                rows.append(
                    {
                        "sensor": sensor,
                        "detector": detector_name,
                        "start": start,
                        "end": end,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["sensor", "detector", "start", "end"])
    table = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    return table
