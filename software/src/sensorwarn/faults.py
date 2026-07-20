"""Fehlerinjektion mit Ground-Truth-Labels.

Jede `inject_*`-Funktion verändert den DataFrame **in-place-kompatibel**
(gibt eine Kopie zurück) und liefert zusätzlich einen Eintrag für das
Label-Log zurück. Ohne dieses Label-Log lässt sich später keine
Precision/Recall-Auswertung (Kapitel 5 der DA) rechnen - es ist also
Pflicht, nicht optional.

Label-Schema (ein Eintrag pro injiziertem Fehler):
    start_time, end_time, sensor, fault_type, magnitude
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FaultLabel:
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    sensor: str
    fault_type: str
    magnitude: float = 0.0


class FaultInjector:
    """Sammelt Fehlerinjektionen + Ground-Truth-Labels für einen Datensatz."""

    def __init__(self, df: pd.DataFrame, seed: int | None = None):
        self.df = df.copy()
        self.labels: list[FaultLabel] = []
        self.rng = np.random.default_rng(seed)

    # -- einzelne Fehlerbilder -------------------------------------------------

    def inject_spike(self, sensor: str, start_idx: int, duration: int = 20, magnitude: float = -10.0) -> "FaultInjector":
        """Kurzer, abrupter Ausreißer (z.B. Messfehler, Störimpuls)."""
        end_idx = min(start_idx + duration, len(self.df))
        self.df.iloc[start_idx:end_idx, self.df.columns.get_loc(sensor)] += magnitude
        self._label(sensor, start_idx, end_idx, "spike", magnitude)
        return self

    def inject_drift(self, sensor: str, start_idx: int, duration: int = 200, total_drift: float = 8.0) -> "FaultInjector":
        """Langsam anwachsende Abweichung (z.B. Sensor-Alterung, Verschmutzung)."""
        end_idx = min(start_idx + duration, len(self.df))
        n = end_idx - start_idx
        ramp = np.linspace(0, total_drift, n)
        col = self.df.columns.get_loc(sensor)
        self.df.iloc[start_idx:end_idx, col] += ramp
        # Drift bleibt nach dem Intervall bestehen (bleibender Versatz), bis Ende der Serie
        self.df.iloc[end_idx:, col] += total_drift
        self._label(sensor, start_idx, end_idx, "drift", total_drift)
        return self

    def inject_frozen(self, sensor: str, start_idx: int, duration: int = 30) -> "FaultInjector":
        """Sensor liefert konstant denselben Wert (eingefroren, kein Rauschen mehr)."""
        end_idx = min(start_idx + duration, len(self.df))
        col = self.df.columns.get_loc(sensor)
        frozen_value = self.df.iloc[start_idx, col]
        self.df.iloc[start_idx:end_idx, col] = frozen_value
        self._label(sensor, start_idx, end_idx, "frozen", 0.0)
        return self

    def inject_dropout(self, sensor: str, start_idx: int, duration: int = 15) -> "FaultInjector":
        """Sensor liefert keine Daten (NaN) - z.B. Kommunikationsausfall."""
        end_idx = min(start_idx + duration, len(self.df))
        col = self.df.columns.get_loc(sensor)
        self.df.iloc[start_idx:end_idx, col] = np.nan
        self._label(sensor, start_idx, end_idx, "dropout", 0.0)
        return self

    def inject_correlated_fault(
        self, sensors: list[str], start_idx: int, duration: int = 25, magnitude: float = 5.0
    ) -> "FaultInjector":
        """Gleichzeitiger Fehler auf mehreren Sensoren (z.B. gemeinsame Ursache,
        etwa Turbinen-Anomalie, die flow, rpm und vib gleichzeitig beeinflusst)."""
        end_idx = min(start_idx + duration, len(self.df))
        for sensor in sensors:
            col = self.df.columns.get_loc(sensor)
            self.df.iloc[start_idx:end_idx, col] += magnitude
            self._label(sensor, start_idx, end_idx, "correlated", magnitude)
        return self

    # -- intern -----------------------------------------------------------------

    def _label(self, sensor: str, start_idx: int, end_idx: int, fault_type: str, magnitude: float) -> None:
        self.labels.append(
            FaultLabel(
                start_time=self.df.index[start_idx],
                end_time=self.df.index[end_idx - 1],
                sensor=sensor,
                fault_type=fault_type,
                magnitude=magnitude,
            )
        )

    def labels_df(self) -> pd.DataFrame:
        return pd.DataFrame([l.__dict__ for l in self.labels])

    def result(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Gibt (fehlerbehafteter DataFrame, Labels-DataFrame) zurück."""
        return self.df, self.labels_df()

    def label_mask(self, sensor: str) -> pd.Series:
        """Boolesche Ground-Truth-Maske je Zeitstempel für einen Sensor -
        praktisch zum direkten Vergleich mit Detektor-Alarmen."""
        mask = pd.Series(False, index=self.df.index)
        for l in self.labels:
            if l.sensor == sensor:
                mask.loc[l.start_time : l.end_time] = True
        return mask
