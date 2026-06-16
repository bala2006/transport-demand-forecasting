import polars as pl
import pytest

from src.data.preprocess import filter_trips, aggregate_hourly_demand, generate_synthetic_data


def test_filter_trips_keeps_required_columns():
    df = pl.DataFrame({
        "tpep_pickup_datetime": ["2024-01-01 00:15:00", "2024-01-01 01:30:00"],
        "PULocationID": [1, 2],
        "extra_col": ["a", "b"],
    })
    result = filter_trips(df)
    assert "pickup_datetime" in result.columns
    assert "PULocationID" in result.columns
    assert "extra_col" not in result.columns


def test_filter_trips_missing_column_raises():
    df = pl.DataFrame({"tpep_pickup_datetime": ["2024-01-01 00:15:00"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        filter_trips(df)


def test_aggregate_hourly_demand_counts_correctly():
    df = pl.DataFrame({
        "pickup_datetime": [
            "2024-01-01 00:15:00",
            "2024-01-01 00:45:00",
            "2024-01-01 01:10:00",
        ],
        "PULocationID": [1, 1, 2],
    }).with_columns(pl.col("pickup_datetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))
    result = aggregate_hourly_demand(df)
    assert result.filter(pl.col("PULocationID") == 1).select("demand").item() == 2
    assert result.filter(pl.col("PULocationID") == 2).select("demand").item() == 1


def test_generate_synthetic_data_has_correct_shape():
    df = generate_synthetic_data(n_zones=5, n_hours=100)
    assert df.height == 5 * 100
    assert "pickup_hour" in df.columns
    assert "PULocationID" in df.columns
    assert "demand" in df.columns
    assert df["demand"].min() >= 0
