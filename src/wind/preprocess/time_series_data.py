from datetime import datetime

import numpy as np
import polars as pl

bidding_area = "ELSPOT NO3"
train_start = datetime(2023, 1, 1)
val_start = datetime(2025, 1, 1)

TAU = 2 * np.pi
hour = pl.col("time").dt.hour().cast(pl.Float64)
doy = pl.col("time").dt.ordinal_day().cast(pl.Float64)
hod_frac = hour / 24.0
doy_frac = (doy - 1 + hour / 24.0) / 365.2425

num_lagged_times = 24
ewma_spans = [3, 6, 12, 24]

ts_data = (
    pl.scan_parquet("../data/wind_power_per_bidzone.parquet")
    .select(time="__index_level_0__", power=bidding_area)
    .sort("time")
    .with_columns(
        sin_hod=(TAU * hour / 24.0).sin(),
        cos_hod=(TAU * hour / 24.0).cos(),
        sin_doy=(TAU * doy_frac).sin(),
        cos_doy=(TAU * doy_frac).cos(),
    )
    .with_columns(
        pl.col("power").shift(k).alias(f"lag_{k:02d}")
        for k in range(1, num_lagged_times + 1)
    )
    .with_columns(
        pl.col("lag_01").ewm_mean(span=span).alias(f"emwa_{span:02d}")
        for span in ewma_spans
    )
    .filter(pl.col("time") >= train_start)
)
