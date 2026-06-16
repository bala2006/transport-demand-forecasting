import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import polars as pl


TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"


def download_parquet(year: int, month: int, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
    dest_path = dest_dir / filename
    if dest_path.exists():
        return dest_path
    url = TLC_BASE_URL.format(year=year, month=month)
    print(f"Downloading {url}...")
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    dest_path.write_bytes(resp.content)
    print(f"Saved {dest_path}")
    return dest_path


def download_tlc_data(years: list[int], months: list[int], dest_dir: Path) -> list[Path]:
    dest_dir = Path(dest_dir)
    tasks = [(y, m) for y in years for m in months]
    paths = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        fut_to_ym = {pool.submit(download_parquet, y, m, dest_dir): (y, m) for y, m in tasks}
        for fut in as_completed(fut_to_ym):
            paths.append(fut.result())
    return sorted(paths)


def download_zone_lookup(dest_dir: Path) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "taxi_zone_lookup.csv"
    if dest_path.exists():
        return dest_path
    print(f"Downloading zone lookup from {ZONE_LOOKUP_URL}...")
    resp = requests.get(ZONE_LOOKUP_URL, timeout=120)
    resp.raise_for_status()
    dest_path.write_text(resp.text)
    print(f"Saved {dest_path}")
    return dest_path


def load_zone_lookup(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, null_values=["", "NA", "N/A"])


def read_trip_parquet(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)
