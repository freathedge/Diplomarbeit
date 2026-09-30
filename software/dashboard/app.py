"""Streamlit-Dashboard: Frühwarnsystem für Sensorfehler im Wasserkraftwerk.

Start:
    cd software
    streamlit run dashboard/app.py

Reine Anzeige-/Bedien-Schicht - alle Berechnungen laufen über
`sensorwarn.dashboard_logic`, damit sie unabhängig von Streamlit
getestet werden können (siehe tests/test_dashboard_logic.py).
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sensorwarn.dashboard_logic import (
    alarm_table,
    combined_alarm,
    compute_sensor_alarms,
    status_light,
)
from sensorwarn.faults import FaultInjector
from sensorwarn.generator import generate_sensor_data

st.set_page_config(page_title="Frühwarnsystem Wasserkraftwerk", page_icon="🌊", layout="wide")

SENSORS = {
    "flow": ("Durchfluss", "m³/s"),
    "press": ("Druck", "bar"),
    "temp": ("Temperatur", "°C"),
    "rpm": ("Drehzahl", "1/min"),
    "vib": ("Vibration", "mm/s"),
}
BLUE, MUTED, GRID = "#3987e5", "#898781", "#2c2c2a"
STATUS = {  # Farbe, Symbol, Text - Status nie nur über Farbe
    "green": ("#0ca30c", "✓", "OK"),
    "yellow": ("#fab219", "!", "Vorwarnung"),
    "red": ("#d03b3b", "✕", "Alarm"),
}

st.markdown(
    """
<style>
.block-container { padding-top: 2.5rem; max-width: 1400px; }
.cards { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:.8rem; margin:1.2rem 0 1.6rem; }
@media (max-width: 1000px) { .cards { grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); } }
.card { background:#1a1a19; border:1px solid #2c2c2a; border-top:3px solid var(--c); border-radius:.7rem; padding:1rem; }
.card .name { color:#c3c2b7; font-size:.85rem; }
.card .value { font-size:1.8rem; font-weight:700; font-variant-numeric:tabular-nums; margin:.3rem 0 .5rem; }
.card .value small { font-size:.85rem; color:#898781; font-weight:500; margin-left:.2rem; }
.badge { display:inline-flex; align-items:center; gap:.35rem; font-size:.75rem; font-weight:600; color:var(--c); }
.badge i { font-style:normal; display:inline-grid; place-items:center; width:1.05rem; height:1.05rem;
  border-radius:50%; background:var(--c); color:#0d0d0d; font-size:.65rem; font-weight:800; }
.card svg { display:block; width:100%; height:34px; margin-top:.6rem; }
</style>
""",
    unsafe_allow_html=True,
)


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


def sparkline(series: pd.Series, n: int = 150) -> str:
    """Mini-Verlauf der letzten `n` Samples als Inline-SVG."""
    values = series.iloc[-n:].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return ""
    lo, span = finite.min(), (finite.max() - finite.min()) or 1.0
    points = " ".join(
        f"{x:.1f},{30 - (v - lo) / span * 26:.1f}"
        for x, v in zip(np.linspace(0, 100, len(values)), values)
        if np.isfinite(v)
    )
    return (
        f'<svg viewBox="0 0 100 32" preserveAspectRatio="none"><polyline points="{points}" fill="none" '
        f'stroke="{BLUE}" stroke-width="1.5" vector-effect="non-scaling-stroke"/></svg>'
    )


def badge(status: str) -> str:
    color, icon, text = STATUS[status]
    return f'<span class="badge" style="--c:{color}"><i>{icon}</i>{text}</span>'


# -- Seitenleiste --------------------------------------------------------------

with st.sidebar:
    st.header("Einstellungen")
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

    n_show = st.slider("Zeitfenster (letzte N Samples)", 200, len(data), min(2000, len(data)), step=100)

    with st.expander("Erweitert"):
        recent_window = st.slider("Alarm = Treffer in letzten N Samples", 5, 200, 30)
        lookback_window = st.slider("Vorwarnung = Treffer in letzten N Samples", 50, 1000, 300)
        zscore_threshold = st.slider("z-Score Schwelle (σ)", 1.0, 6.0, 3.0)
        ma_threshold = st.slider("Moving-Average Schwelle (σ)", 1.0, 6.0, 3.0)

params = {"zscore": {"threshold": zscore_threshold}, "moving_average": {"threshold": ma_threshold}}
view = data.iloc[-n_show:]
sensors = list(data.columns)
alarms = {s: compute_sensor_alarms(view, s, params=params) for s in sensors}
combined = {s: combined_alarm(alarms[s]) for s in sensors}
status = {s: status_light(combined[s], recent_window=recent_window, lookback_window=lookback_window) for s in sensors}

# -- Kopfzeile -------------------------------------------------------------------

n_red = sum(v == "red" for v in status.values())
n_yellow = sum(v == "yellow" for v in status.values())
overall = "red" if n_red else "yellow" if n_yellow else "green"
summary = {
    "red": f"{n_red} Sensor(en) im Alarm",
    "yellow": f"{n_yellow} Sensor(en) mit Vorwarnung",
    "green": "Alle Sensoren im Normalbereich",
}[overall]

st.title("Frühwarnsystem Wasserkraftwerk")
st.markdown(
    f'{badge(overall)} &nbsp;<span style="color:#c3c2b7">{summary} · Stand {view.index[-1]:%d.%m.%Y %H:%M}</span>',
    unsafe_allow_html=True,
)

# -- Sensor-Karten -----------------------------------------------------------------

cards = []
for s in sensors:
    name, unit = SENSORS.get(s, (s, ""))
    last = view[s].iloc[-1]
    value = "—" if pd.isna(last) else f"{last:.1f}" if abs(last) >= 100 else f"{last:.2f}"
    cards.append(
        f'<div class="card" style="--c:{STATUS[status[s]][0]}">'
        f'<div class="name">{name}</div>'
        f'<div class="value">{value}<small>{unit}</small></div>'
        f"{badge(status[s])}{sparkline(view[s])}</div>"
    )
st.markdown(f'<div class="cards">{"".join(cards)}</div>', unsafe_allow_html=True)

# -- Verlauf eines Sensors -------------------------------------------------------------

sensor = st.segmented_control(
    "Verlauf", sensors, default=sensors[0], required=True, format_func=lambda s: SENSORS.get(s, (s, ""))[0]
)
name, unit = SENSORS.get(sensor, (sensor, ""))
frame = pd.DataFrame({"Zeit": view.index, "Wert": view[sensor].to_numpy(), "Alarm": combined[sensor].to_numpy()})

x = alt.X("Zeit:T", title=None, axis=alt.Axis(format="%d.%m. %H:%M", grid=False))
y = alt.Y("Wert:Q", title=unit, scale=alt.Scale(zero=False))
hover = alt.selection_point(nearest=True, on="pointermove", fields=["Zeit"], empty=False, clear="pointerout")
base = alt.Chart(frame).encode(x=x, y=y)

layers = [
    base.mark_point(opacity=0).encode(
        tooltip=[alt.Tooltip("Zeit:T", format="%d.%m.%Y %H:%M"), alt.Tooltip("Wert:Q", format=".2f", title=f"{name} [{unit}]")]
    ).add_params(hover),
    base.mark_line(color=BLUE, strokeWidth=1.5),
    base.transform_filter("datum.Alarm").mark_circle(color=STATUS["red"][0], size=45, opacity=0.9),
    base.mark_circle(size=80, color=BLUE, stroke="white", strokeWidth=1.5).encode(
        opacity=alt.when(hover).then(alt.value(1)).otherwise(alt.value(0))
    ),
]
if labels is not None:
    faults = labels[(labels["sensor"] == sensor) & (labels["end_time"] >= view.index[0])]
    if not faults.empty:
        layers.insert(1, alt.Chart(faults).mark_rect(color=MUTED, opacity=0.18).encode(x="start_time:T", x2="end_time:T"))

chart = (
    alt.layer(*layers)
    .properties(height=380)
    .configure(background="transparent")
    .configure_view(stroke=None)
    .configure_axis(gridColor=GRID, domainColor=GRID, tickColor=GRID, labelColor=MUTED, titleColor=MUTED)
)
st.altair_chart(chart, theme=None, width="stretch")
st.caption("Rote Punkte = Alarm" + (" · graue Bereiche = eingebaute Testfehler" if labels is not None else ""))

# -- Alarmliste ---------------------------------------------------------------------

table = alarm_table(alarms)
with st.expander(f"Alarmliste ({len(table)} Einträge)"):
    if table.empty:
        st.write("Keine Alarme im Zeitfenster.")
    else:
        st.dataframe(table.sort_values("start", ascending=False), hide_index=True, width="stretch")
        st.download_button("Als CSV exportieren", table.to_csv(index=False).encode("utf-8"), "alarme.csv", "text/csv")
