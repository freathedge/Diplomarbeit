"""Erzeugt ein Beispiel-Szenario mit mehreren Fehlerbildern und speichert
Rohdaten + Ground-Truth-Labels als CSV (data/).

Aufruf:
    python scripts/generate_demo_dataset.py
"""
from pathlib import Path

from sensorwarn.faults import FaultInjector
from sensorwarn.generator import generate_sensor_data

OUT_DIR = Path(__file__).resolve().parent.parent / "data"


def build_demo_scenario():
    df = generate_sensor_data(n_samples=5000, seed=42)
    inj = FaultInjector(df, seed=42)

    inj.inject_spike("flow", start_idx=1000, duration=20, magnitude=-10.0)
    inj.inject_spike("press", start_idx=1500, duration=15, magnitude=1.5)
    inj.inject_drift("flow", start_idx=2000, duration=300, total_drift=8.0)
    inj.inject_drift("press", start_idx=2600, duration=250, total_drift=1.2)
    inj.inject_frozen("press", start_idx=3200, duration=40)
    inj.inject_frozen("vib", start_idx=3400, duration=40)
    inj.inject_dropout("flow", start_idx=3800, duration=20)
    inj.inject_dropout("temp", start_idx=4000, duration=20)
    inj.inject_correlated_fault(["flow", "rpm", "vib"], start_idx=4400, duration=30, magnitude=6.0)

    return inj.result()


def main():
    OUT_DIR.mkdir(exist_ok=True)
    data, labels = build_demo_scenario()

    data_path = OUT_DIR / "demo_sensor_data.csv"
    labels_path = OUT_DIR / "demo_labels.csv"
    data.to_csv(data_path)
    labels.to_csv(labels_path, index=False)

    print(f"Datenpunkte: {len(data)}  ->  {data_path}")
    print(f"Injizierte Fehler: {len(labels)}  ->  {labels_path}")
    print(labels.to_string(index=False))


if __name__ == "__main__":
    main()
