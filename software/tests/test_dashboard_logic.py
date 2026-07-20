import pandas as pd

from sensorwarn.dashboard_logic import (
    alarm_table,
    combined_alarm,
    compute_sensor_alarms,
    status_light,
)
from sensorwarn.faults import FaultInjector
from sensorwarn.generator import generate_sensor_data


def _base_df():
    return generate_sensor_data(n_samples=2000, seed=1)


def test_clean_data_status_is_never_red():
    """Fünf Detektoren werden per OR kombiniert (bewusst hohe Sensitivität,
    siehe Projektnotiz "lieber zu viel prüfen") - dadurch kann auf sauberen
    Daten gelegentlich ein vereinzelter False Positive ein "gelb" auslösen.
    Ein anhaltendes "rot" (Alarm in den letzten `recent_window` Samples)
    darf auf fehlerfreien Daten aber nicht auftreten."""
    df = _base_df()
    alarms = compute_sensor_alarms(df, "flow")
    combined = combined_alarm(alarms)
    assert status_light(combined) != "red"


def test_recent_fault_gives_red_status():
    df = _base_df()
    inj = FaultInjector(df, seed=1)
    inj.inject_dropout("temp", start_idx=len(df) - 20, duration=15)
    result, _ = inj.result()

    alarms = compute_sensor_alarms(result, "temp")
    combined = combined_alarm(alarms)
    assert status_light(combined) == "red"


def test_older_fault_gives_yellow_not_red():
    df = _base_df()
    inj = FaultInjector(df, seed=1)
    # Fehler liegt weit genug zurück, dass er nicht mehr "akut" (recent_window) ist,
    # aber noch im lookback_window liegt.
    inj.inject_dropout("temp", start_idx=len(df) - 100, duration=15)
    result, _ = inj.result()

    alarms = compute_sensor_alarms(result, "temp")
    combined = combined_alarm(alarms)
    assert status_light(combined, recent_window=30, lookback_window=300) == "yellow"


def test_alarm_table_contains_injected_fault():
    df = _base_df()
    inj = FaultInjector(df, seed=1)
    inj.inject_dropout("temp", start_idx=500, duration=15)
    result, _ = inj.result()

    alarms = {"temp": compute_sensor_alarms(result, "temp")}
    table = alarm_table(alarms)

    dropout_rows = table[(table["sensor"] == "temp") & (table["detector"] == "dropout")]
    assert len(dropout_rows) == 1
    assert dropout_rows.iloc[0]["start"] == result.index[500]
    assert dropout_rows.iloc[0]["end"] == result.index[514]


def test_alarm_table_empty_when_no_alarms():
    df = _base_df()
    alarms = {"press": compute_sensor_alarms(df, "press")}
    table = alarm_table(alarms)
    # sauberer Datensatz: frozen/dropout sollten still bleiben
    assert not ((table["sensor"] == "press") & (table["detector"] == "dropout")).any()
    assert not ((table["sensor"] == "press") & (table["detector"] == "frozen")).any()
