"""Streamlit-Dashboard: Frühwarnsystem für Sensorfehler im Wasserkraftwerk.

Start:
    cd software
    streamlit run dashboard/app.py

Reine Anzeige-/Bedien-Schicht - alle Berechnungen laufen über
`sensorwarn.dashboard_logic`, damit sie unabhängig von Streamlit
getestet werden können (siehe tests/test_dashboard_logic.py).
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sensorwarn.dashboard_logic import (
    STATUS_EMOJI,
    alarm_table,
    combined_alarm,
    compute_sensor_alarms,
    status_light,
)
from sensorwarn.faults import FaultInjector
from sensorwarn.generator import generate_sensor_data

st.set_page_config(page_title="Frühwarnsystem Wasserkraftwerk", layout="wide")

SENSOR_UNITS = {"flow": "m3/s", "press": "bar", "temp": "°C", "rpm": "1/min", "vib": "mm/s"}


@st.cache_data
def build_demo_dataset(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Erzeugt denselben Demo-Datensatz wie scripts/generate_demo_dataset.py,
    damit im Dashboard und in der Auswertung dieselben Fehlerbilder auftauchen."""
    df = generate_sensor_data(n_samples=5000, seed=seed)
    inj = FaultInjector(df, seed=seed)
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


st.title("🌊 Digitales Frühwarnsystem – Sensorfehler im Wasserkraftwerk")
st.caption("Prototyp · Diplomarbeit Buder/Frick · Auftraggeber illwerke vkw AG")

with st.sidebar:
    st.header("Daten")
    source = st.radio("Datenquelle", ["Demo-Szenario", "Eigene CSV hochladen"])

    if source == "Demo-Szenario":
        seed = st.number_input("Seed", value=42, step=1)
        data, labels = build_demo_dataset(int(seed))
    else:
        upload = st.file_uploader("Sensordaten (CSV, Spalte 'timestamp' als Index)", type="csv")
        if upload is None:
            st.info("Bitte CSV hochladen oder Demo-Szenario wählen.")
            st.stop()
        data = pd.read_csv(upload, index_col="timestamp", parse_dates=True)
        labels = None

    st.header("Ansicht")
    n_show = st.slider("Anzuzeigende Samples (letzte N)", 200, len(data), min(2000, len(data)))
    recent_window = st.slider("Ampel: 'akut' = letzte N Samples", 5, 200, 30)
    lookback_window = st.slider("Ampel: 'Vorwarnung' = letzte N Samples", 50, 1000, 300)

    st.header("Detektor-Parameter")
    zscore_window = st.slider("z-Score Fenster", 10, 200, 60)
    zscore_threshold = st.slider("z-Score Schwelle (σ)", 1.0, 6.0, 3.0)
    ma_window = st.slider("Moving-Average Fenster", 10, 200, 30)
    ma_threshold = st.slider("Moving-Average Schwelle (σ)", 1.0, 6.0, 3.0)

params = {
    "zscore": {"window": zscore_window, "threshold": zscore_threshold},
    "moving_average": {"window": ma_window, "threshold": ma_threshold},
}

view = data.iloc[-n_show:]
sensors = list(data.columns)

all_alarms = {}
for sensor in sensors:
    all_alarms[sensor] = compute_sensor_alarms(view, sensor, params=params)

# -- Status-Ampeln -----------------------------------------------------------

st.subheader("Status")
cols = st.columns(len(sensors))
for col, sensor in zip(cols, sensors):
    combined = combined_alarm(all_alarms[sensor])
    status = status_light(combined, recent_window=recent_window, lookback_window=lookback_window)
    unit = SENSOR_UNITS.get(sensor, "")
    col.metric(label=f"{STATUS_EMOJI[status]} {sensor} [{unit}]", value=status.upper())

# -- Zeitreihen mit markierten Anomalien -------------------------------------

st.subheader("Zeitreihen")
for sensor in sensors:
    combined = combined_alarm(all_alarms[sensor])
    fig, ax = plt.subplots(figsize=(12, 2.2))
    ax.plot(view.index, view[sensor], linewidth=0.8, color="#1f77b4")
    alarm_points = view[sensor][combined]
    if not alarm_points.empty:
        ax.scatter(alarm_points.index, alarm_points.values, color="red", s=10, zorder=3)
    ax.set_title(sensor)
    ax.set_ylabel(SENSOR_UNITS.get(sensor, ""))
    st.pyplot(fig)
    plt.close(fig)

# -- Alarmtabelle -------------------------------------------------------------

st.subheader("Alarme")
table = alarm_table(all_alarms)
st.dataframe(table, width="stretch")

if not table.empty:
    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button("Alarme als CSV exportieren", csv, "alarme.csv", "text/csv")

if labels is not None:
    with st.expander("Ground-Truth-Labels (injizierte Fehler, Demo-Szenario)"):
        st.dataframe(labels, width="stretch")
