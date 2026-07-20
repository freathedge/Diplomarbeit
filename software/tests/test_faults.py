import numpy as np

from sensorwarn.faults import FaultInjector
from sensorwarn.generator import generate_sensor_data


def _base_df():
    return generate_sensor_data(n_samples=1000, seed=1)


def test_spike_changes_values_and_labels():
    df = _base_df()
    inj = FaultInjector(df, seed=1)
    inj.inject_spike("flow", start_idx=500, duration=10, magnitude=-10.0)
    result, labels = inj.result()

    assert not np.allclose(result["flow"].iloc[500:510].values, df["flow"].iloc[500:510].values)
    assert len(labels) == 1
    assert labels.iloc[0]["fault_type"] == "spike"
    assert labels.iloc[0]["sensor"] == "flow"


def test_dropout_produces_nan():
    df = _base_df()
    inj = FaultInjector(df, seed=1)
    inj.inject_dropout("temp", start_idx=100, duration=15)
    result, labels = inj.result()

    assert result["temp"].iloc[100:115].isna().all()
    assert labels.iloc[0]["fault_type"] == "dropout"


def test_frozen_removes_variance():
    df = _base_df()
    inj = FaultInjector(df, seed=1)
    inj.inject_frozen("press", start_idx=200, duration=30)
    result, _ = inj.result()

    assert result["press"].iloc[200:230].std() < 1e-9  # exakt konstant (Rest ist Gleitkomma-Rauschen)


def test_correlated_fault_hits_all_sensors():
    df = _base_df()
    inj = FaultInjector(df, seed=1)
    inj.inject_correlated_fault(["flow", "rpm", "vib"], start_idx=300, duration=20, magnitude=5.0)
    _, labels = inj.result()

    assert set(labels["sensor"]) == {"flow", "rpm", "vib"}
    assert (labels["fault_type"] == "correlated").all()


def test_label_mask_matches_injected_interval():
    df = _base_df()
    inj = FaultInjector(df, seed=1)
    inj.inject_spike("flow", start_idx=500, duration=10, magnitude=-10.0)
    mask = inj.label_mask("flow")

    assert mask.sum() == 10
    assert mask.iloc[500:510].all()
    assert not mask.iloc[:500].any()
