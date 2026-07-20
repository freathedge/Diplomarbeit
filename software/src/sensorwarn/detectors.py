"""Detektionsalgorithmen für Sensoranomalien.

Alle Detektoren folgen derselben Schnittstelle:

    detect(series: pd.Series, **params) -> pd.Series[bool]

Rückgabe ist eine boolesche Alarm-Maske mit demselben Index wie die
Eingabe. Dadurch sind Detektoren in der Evaluation (metrics.py) und im
Dashboard austauschbar, ohne Sonderfälle je Algorithmus zu brauchen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def moving_average_detector(series: pd.Series, window: int = 30, threshold: float = 3.0) -> pd.Series:
    """Alarm, wenn Abweichung vom gleitenden Mittel mehr als `threshold`
    Standardabweichungen (über dasselbe Fenster) beträgt."""
    rolling_mean = series.rolling(window, min_periods=window // 2).mean()
    rolling_std = series.rolling(window, min_periods=window // 2).std()
    deviation = (series - rolling_mean).abs()
    alarm = deviation > (threshold * rolling_std)
    return alarm.fillna(False)


def ewma_detector(series: pd.Series, alpha: float = 0.1, threshold: float = 3.0) -> pd.Series:
    """Wie moving_average_detector, aber mit exponentiell gewichtetem
    Mittel/Std - reagiert schneller auf neue Abweichungen."""
    ewma_mean = series.ewm(alpha=alpha).mean()
    ewma_std = series.ewm(alpha=alpha).std()
    deviation = (series - ewma_mean).abs()
    alarm = deviation > (threshold * ewma_std)
    return alarm.fillna(False)


def zscore_detector(series: pd.Series, window: int = 60, threshold: float = 3.0) -> pd.Series:
    """Rollierender z-Score: (x - Fenster-Mittel) / Fenster-Std."""
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    z = (series - mean) / std.replace(0, np.nan)
    return (z.abs() > threshold).fillna(False)


def frozen_detector(series: pd.Series, window: int = 10, eps: float = 1e-6) -> pd.Series:
    """Alarm, wenn die Varianz über ein rollierendes Fenster praktisch
    null ist - Hinweis auf einen eingefrorenen Sensor."""
    rolling_std = series.rolling(window, min_periods=window).std()
    return (rolling_std < eps).fillna(False)


def dropout_detector(series: pd.Series) -> pd.Series:
    """Alarm bei fehlenden Werten (NaN) - z.B. Kommunikationsausfall."""
    return series.isna()


def drift_detector(series: pd.Series, window: int = 120, slope_threshold: float = 0.02) -> pd.Series:
    """Alarm, wenn die lineare Steigung über ein längeres Rollfenster
    einen Schwellenwert überschreitet - typisch für langsame Drift."""
    x = np.arange(window)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return 0.0
        return float(((x - x_mean) * (values - values.mean())).sum() / x_var)

    slopes = series.rolling(window).apply(_slope, raw=True)
    return (slopes.abs() > slope_threshold).fillna(False)


def pca_detector(
    df: pd.DataFrame,
    sensors: list[str] | None = None,
    n_components: int = 1,
    quantile: float = 0.99,
    reference: pd.DataFrame | None = None,
) -> pd.Series:
    """Multivariater Detektor: PCA auf allen Sensoren, Alarm bei hohem
    Rekonstruktionsfehler (Q-Statistik). Erkennt Anomalien, die sich nur
    in der *Kombination* mehrerer Sensoren zeigen (korrelierte Fehler).

    `n_components` bewusst klein (Standard: 1) gewählt: je mehr Komponenten,
    desto mehr saugt die PCA auch die anomale Richtung als "normale"
    Varianz auf und der Fehler verschwindet im Rekonstruktionsfehler.

    `reference`: optionaler, bekannt sauberer Referenzdatensatz zum Fitten
    von Mittelwert/Std/PCA-Modell. Ohne Referenz wird auf `df` selbst
    gefittet (praktisch für Exploration, im Betrieb sollte immer auf
    historischen Normaldaten kalibriert werden, sonst kann eine große
    Anomalie im gleichen Fenster die Schwelle mit hochziehen).

    Die Q-Statistik ist rechtsschief (näherungsweise Chi-Quadrat-verteilt),
    daher wird die Schwelle über ein Quantil statt über Mittelwert+Std
    bestimmt.
    """
    sensors = sensors or list(df.columns)
    fit_data = (reference if reference is not None else df)[sensors].ffill().bfill()
    data = df[sensors].ffill().bfill()

    mean = fit_data.mean()
    std = fit_data.std().replace(0, 1.0)

    pca = PCA(n_components=n_components)
    pca.fit((fit_data - mean) / std)

    data_norm = (data - mean) / std
    scores = pca.transform(data_norm)
    reconstructed = pca.inverse_transform(scores)
    residual = data_norm.values - reconstructed
    q_stat = pd.Series(np.sum(residual**2, axis=1), index=df.index)

    threshold = q_stat.quantile(quantile)
    return q_stat > threshold


DETECTORS = {
    "moving_average": moving_average_detector,
    "ewma": ewma_detector,
    "zscore": zscore_detector,
    "frozen": frozen_detector,
    "dropout": dropout_detector,
    "drift": drift_detector,
}
