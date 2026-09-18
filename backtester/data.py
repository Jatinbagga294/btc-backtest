"""Loading the processed feature files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def load(timeframe: str = "1d", data_dir: Path | None = None) -> pd.DataFrame:
    """Load one timeframe's feature file, indexed by timestamp."""
    directory = data_dir or DATA_DIR
    path = directory / f"btc_{timeframe}_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Market data is not committed to this repo. "
            f"See data/README.md for how to rebuild it from Binance."
        )
    df = pd.read_csv(path, parse_dates=["open_time"]).set_index("open_time")
    return df.dropna(subset=["close"])
