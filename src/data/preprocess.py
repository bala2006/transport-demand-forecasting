from pathlib import Path
import polars as pl


def filter_trips(df: pl.DataFrame) -> pl.DataFrame:
    required = ["tpep_pickup_datetime", "PULocationID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    filtered = df.with_columns(
        pl.col("tpep_pickup_datetime").alias("pickup_datetime")
    ).filter(
        pl.col("PULocationID").is_not_null()
    ).select([
        "pickup_datetime",
        "PULocationID",
    ])
    return filtered


def aggregate_hourly_demand(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("pickup_datetime").dt.truncate("1h").alias("pickup_hour")
    ).group_by(
        ["pickup_hour", "PULocationID"]
    ).agg(
        pl.len().alias("demand")
    ).sort("pickup_hour", "PULocationID")


def chronological_split(
    df: pl.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df = df.sort("pickup_hour")
    n = df.height
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = df[:train_end]
    val = df[train_end:val_end]
    test = df[val_end:]
    return train, val, test


def run_preprocessing(
    raw_dir: Path,
    processed_dir: Path,
    zone_lookup_path: Path,
    years: list[int],
    months: list[int],
) -> pl.DataFrame:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    import polars as pl
    from src.data.download import read_trip_parquet

    all_frames = []
    for y in years:
        for m in months:
            path = raw_dir / f"yellow_tripdata_{y}-{m:02d}.parquet"
            if path.exists():
                frame = read_trip_parquet(path)
                all_frames.append(frame)
    if not all_frames:
        print("No raw parquet files found. Run download first or use synthetic fallback.")
        return generate_synthetic_data()

    df = pl.concat(all_frames)
    df = filter_trips(df)
    df = aggregate_hourly_demand(df)

    with open(zone_lookup_path) as _f:
        _csv_text = "".join(line for line in _f if line.strip())
    zones = pl.read_csv(_csv_text.encode(), has_header=True, infer_schema_length=10000)
    zone_map = zones.select([
        pl.col("LocationID").alias("PULocationID"),
        pl.col("Zone").alias("pickup_zone"),
        pl.col("Borough").alias("borough"),
    ])
    df = df.join(zone_map, on="PULocationID", how="left")

    output_path = processed_dir / "hourly_demand.parquet"
    df.write_parquet(output_path)
    print(f"Saved aggregated data to {output_path}")
    return df


def generate_synthetic_data(
    n_zones: int = 10,
    n_hours: int = 3000,
) -> pl.DataFrame:
    import datetime

    import numpy as np

    rng = np.random.default_rng(42)
    base_dt = datetime.datetime(2024, 1, 1, 0, 0, 0)
    hours = [base_dt + datetime.timedelta(hours=i) for i in range(n_hours)]
    rows = []
    for h in hours:
        for zone_id in range(1, n_zones + 1):
            demand = max(0, int(rng.poisson(20 + 10 * np.sin(h.hour * 2 * np.pi / 24))))
            rows.append({"pickup_hour": h, "PULocationID": zone_id, "demand": demand})
    df = pl.DataFrame(rows)
    return df
