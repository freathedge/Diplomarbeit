"""Erzeugung synthetischer, aber physikalisch plausibler Sensordaten
für ein Wasserkraftwerk (Durchfluss, Temperatur, Druck, Drehzahl, Vibration).

Die Sensoren sind nicht unabhängig: Drehzahl und Vibration folgen dem
Durchfluss mit etwas Verzögerung/Rauschen, damit korrelierte Fehler
(Task 2 im Aufbau der DA) später sinnvoll simuliert werden können.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SensorConfig:
    """Grundparameter eines einzelnen Sensors."""

    name: str
    base_value: float
    noise_std: float
    daily_amplitude: float = 0.0  # Tagesgang (z.B. Lastprofil)
    unit: str = ""


DEFAULT_SENSORS: dict[str, SensorConfig] = {
    "flow": SensorConfig("flow", base_value=100.0, noise_std=0.3, daily_amplitude=5.0, unit="m3/s"),
    "press": SensorConfig("press", base_value=12.0, noise_std=0.05, daily_amplitude=0.3, unit="bar"),
    "temp": SensorConfig("temp", base_value=25.0, noise_std=0.2, daily_amplitude=1.0, unit="°C"),
    "rpm": SensorConfig("rpm", base_value=500.0, noise_std=1.5, daily_amplitude=15.0, unit="1/min"),
    "vib": SensorConfig("vib", base_value=2.0, noise_std=0.05, daily_amplitude=0.1, unit="mm/s"),
}


def generate_sensor_data(
    n_samples: int = 5000,
    freq: str = "min",
    start: str = "2026-01-01",
    seed: int | None = 42,
    sensors: dict[str, SensorConfig] | None = None,
) -> pd.DataFrame:
    """Erzeugt einen sauberen (fehlerfreien) Basisdatensatz.

    flow ist das treibende Signal. rpm und vib werden daraus mit
    Kopplungsfaktor + eigenem Rauschen abgeleitet, damit ein Fehler in
    `flow` sich real auch in den gekoppelten Sensoren zeigen kann.
    """
    sensors = sensors or DEFAULT_SENSORS
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=n_samples, freq=freq)

    # Tageszyklus in Samples-Einheiten (bei "min"-Frequenz: 1440 Samples/Tag)
    samples_per_day = pd.Timedelta("1D") / pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    phase = np.linspace(0, n_samples / samples_per_day * 2 * np.pi, n_samples)

    flow_cfg = sensors["flow"]
    flow = (
        flow_cfg.base_value
        + flow_cfg.daily_amplitude * np.sin(phase)
        + rng.normal(0, flow_cfg.noise_std, n_samples)
    )

    press_cfg = sensors["press"]
    press = (
        press_cfg.base_value
        + press_cfg.daily_amplitude * np.sin(phase + 0.3)
        + rng.normal(0, press_cfg.noise_std, n_samples)
    )

    temp_cfg = sensors["temp"]
    temp = (
        temp_cfg.base_value
        + temp_cfg.daily_amplitude * np.sin(phase / 10)  # trägere Dynamik
        + rng.normal(0, temp_cfg.noise_std, n_samples)
    )

    # rpm koppelt an flow (mehr Durchfluss -> höhere Drehzahl)
    rpm_cfg = sensors["rpm"]
    flow_norm = (flow - flow_cfg.base_value) / max(flow_cfg.daily_amplitude, 1e-6)
    rpm = rpm_cfg.base_value + rpm_cfg.daily_amplitude * flow_norm + rng.normal(0, rpm_cfg.noise_std, n_samples)

    # vib koppelt an rpm (mechanische Unwucht steigt mit Drehzahl)
    vib_cfg = sensors["vib"]
    rpm_norm = (rpm - rpm_cfg.base_value) / max(rpm_cfg.daily_amplitude, 1e-6)
    vib = vib_cfg.base_value + vib_cfg.daily_amplitude * np.abs(rpm_norm) + rng.normal(0, vib_cfg.noise_std, n_samples)

    df = pd.DataFrame(
        {"flow": flow, "press": press, "temp": temp, "rpm": rpm, "vib": vib},
        index=t,
    )
    df.index.name = "timestamp"
    return df
