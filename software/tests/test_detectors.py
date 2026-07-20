from sensorwarn.detectors import (
    drift_detector,
    dropout_detector,
    ewma_detector,
    frozen_detector,
    moving_average_detector,
    pca_detector,
    zscore_detector,
)
from sensorwarn.faults import FaultInjector
from sensorwarn.generator import generate_sensor_data


def _base_df():
    return generate_sensor_data(n_samples=2000, seed=1)


def test_clean_data_triggers_few_alarms():
    """Negativtest: auf sauberen Daten sollen die meisten Samples ruhig bleiben."""
    df = _base_df()
    for detector in (moving_average_detector, ewma_detector, zscore_detector):
        alarm = detector(df["flow"])
        assert alarm.mean() < 0.05  # < 5% False-Positive-Rate auf sauberen Daten


def test_moving_average_detects_spike():
    df = _base_df()
    inj = FaultInjector(df, seed=1)
    inj.inject_spike("flow", start_idx=1000, duration=10, magnitude=-10.0)
    result, _ = inj.result()

    alarm = moving_average_detector(result["flow"], window=30, threshold=3.0)
    assert alarm.iloc[1000:1010].any()


def test_zscore_detects_spike():
    df = _base_df()
    inj = FaultInjector(df, seed=1)
    inj.inject_spike("flow", start_idx=1000, duration=10, magnitude=-10.0)
    result, _ = inj.result()

    alarm = zscore_detector(result["flow"])
    assert alarm.iloc[1000:1010].any()


def test_frozen_detector_detects_frozen_sensor():
    df = _base_df()
    inj = FaultInjector(df, seed=1)
    inj.inject_frozen("press", start_idx=500, duration=30)
    result, _ = inj.result()

    alarm = frozen_detector(result["press"], window=10)
    assert alarm.iloc[500:530].any()


def test_frozen_detector_quiet_on_clean_data():
    df = _base_df()
    alarm = frozen_detector(df["press"], window=10)
    assert alarm.sum() == 0


def test_dropout_detector_detects_nan():
    df = _base_df()
    inj = FaultInjector(df, seed=1)
    inj.inject_dropout("temp", start_idx=300, duration=15)
    result, _ = inj.result()

    alarm = dropout_detector(result["temp"])
    assert alarm.iloc[300:315].all()
    assert not alarm.iloc[:300].any()


def test_drift_detector_detects_drift():
    df = _base_df()
    inj = FaultInjector(df, seed=1)
    inj.inject_drift("flow", start_idx=500, duration=200, total_drift=15.0)
    result, _ = inj.result()

    alarm = drift_detector(result["flow"], window=120, slope_threshold=0.02)
    assert alarm.iloc[500:700].any()


def test_pca_detector_detects_correlated_fault():
    df = _base_df()
    inj = FaultInjector(df, seed=1)
    inj.inject_correlated_fault(["flow", "rpm", "vib"], start_idx=1000, duration=25, magnitude=8.0)
    result, _ = inj.result()

    alarm = pca_detector(result, sensors=["flow", "press", "temp", "rpm", "vib"])
    assert alarm.iloc[1000:1025].any()
