import pandas as pd

from sensorwarn.generator import DEFAULT_SENSORS, generate_sensor_data


def test_generate_sensor_data_shape():
    df = generate_sensor_data(n_samples=500, seed=1)
    assert len(df) == 500
    assert list(df.columns) == list(DEFAULT_SENSORS.keys())
    assert isinstance(df.index, pd.DatetimeIndex)


def test_generate_sensor_data_no_nans_by_default():
    df = generate_sensor_data(n_samples=1000, seed=1)
    assert not df.isna().any().any()


def test_generate_sensor_data_reproducible_with_seed():
    df1 = generate_sensor_data(n_samples=200, seed=7)
    df2 = generate_sensor_data(n_samples=200, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_sensor_data_differs_with_different_seed():
    df1 = generate_sensor_data(n_samples=200, seed=1)
    df2 = generate_sensor_data(n_samples=200, seed=2)
    assert not df1["flow"].equals(df2["flow"])


def test_rpm_correlates_with_flow():
    df = generate_sensor_data(n_samples=3000, seed=1)
    corr = df["flow"].corr(df["rpm"])
    assert corr > 0.5
