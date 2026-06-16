import polars as pl
import numpy as np


US_FEDERAL_HOLIDAYS_2024 = [
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-10-14",
    "2024-11-11", "2024-11-28", "2024-12-25",
]


def create_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("pickup_hour").dt.hour().alias("hour"),
        pl.col("pickup_hour").dt.weekday().alias("day_of_week"),
        pl.col("pickup_hour").dt.month().alias("month"),
        (pl.col("pickup_hour").dt.weekday().is_in([6, 7])).alias("is_weekend"),
        (pl.col("pickup_hour").dt.hour() * 2 * np.pi / 24).sin().alias("hour_sin"),
        (pl.col("pickup_hour").dt.hour() * 2 * np.pi / 24).cos().alias("hour_cos"),
        ((pl.col("pickup_hour").dt.weekday() - 1) * 2 * np.pi / 7).sin().alias("dow_sin"),
        ((pl.col("pickup_hour").dt.weekday() - 1) * 2 * np.pi / 7).cos().alias("dow_cos"),
        ((pl.col("pickup_hour").dt.month() - 1) * 2 * np.pi / 12).sin().alias("month_sin"),
        ((pl.col("pickup_hour").dt.month() - 1) * 2 * np.pi / 12).cos().alias("month_cos"),
    )


def add_holiday_features(df: pl.DataFrame) -> pl.DataFrame:
    holiday_dates = pl.Series("_holiday_dates", US_FEDERAL_HOLIDAYS_2024).cast(pl.Date)
    return df.with_columns(
        pl.col("pickup_hour").dt.date().is_in(holiday_dates).alias("is_holiday"),
        (pl.col("pickup_hour").dt.date() - pl.duration(days=1)).is_in(holiday_dates).alias("day_after_holiday"),
    )


def add_lag_features(df: pl.DataFrame, group_col: str = "PULocationID") -> pl.DataFrame:
    df = df.sort("pickup_hour")
    for lag in [1, 2, 3, 24]:
        df = df.with_columns(
            pl.col("demand").shift(lag).over(group_col).alias(f"demand_lag_{lag}h")
        )
    for window in [1, 3, 6]:
        df = df.with_columns(
            pl.col("demand").rolling_mean(window_size=window).shift(1).over(group_col).alias(f"rolling_mean_{window}h")
        )
    return df


def add_weather_features(df: pl.DataFrame, weather_path: str | None = None) -> pl.DataFrame:
    if weather_path is None:
        return df.with_columns(
            pl.lit(20.0).alias("temp_c"),
            pl.lit(0.0).alias("precipitation_mm"),
            pl.lit(10.0).alias("wind_speed_kph"),
            pl.lit(0).alias("snow_flag"),
        )
    weather = pl.read_csv(weather_path, try_parse_dates=True)
    weather = weather.with_columns(
        pl.col("DATE").cast(pl.Date).alias("weather_date"),
        pl.col("TEMP").cast(pl.Float64).alias("temp_c"),
        pl.col("PRCP").cast(pl.Float64).alias("precipitation_mm"),
        pl.col("WDSP").cast(pl.Float64).alias("wind_speed_kph"),
        pl.col("SNOW").cast(pl.Float64).gt(0).alias("snow_flag"),
    ).select(["weather_date", "temp_c", "precipitation_mm", "wind_speed_kph", "snow_flag"])
    df = df.with_columns(pl.col("pickup_hour").dt.date().alias("weather_date"))
    df = df.join(weather, on="weather_date", how="left")
    return df.drop("weather_date")


def build_feature_matrix(df: pl.DataFrame, weather_path: str | None = None) -> pl.DataFrame:
    df = create_temporal_features(df)
    df = add_holiday_features(df)
    df = add_lag_features(df)
    df = add_weather_features(df, weather_path)
    df = df.drop_nulls()
    return df
