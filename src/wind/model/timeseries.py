from datetime import datetime

import numpy as np
import polars as pl


def fill_gaps(df: pl.LazyFrame, time_col: str, interval: str = "1h") -> pl.LazyFrame:
    times = df.select(
        pl.datetime_range(pl.col(time_col).min(), pl.col(time_col).max(), interval)
    )
    filled = times.join(df, on=time_col, how="left").fill_null(strategy="forward")
    return filled


def get_timeseries_dataset(
    lookback: int, forecast_lead_time: int, forecast_window: int
):
    windpower = (
        pl.scan_parquet("data/wind_power_per_bidzone.parquet")
        .rename({"__index_level_0__": "time"})
        .filter(pl.col("time") >= datetime(2021, 1, 1))
        .sort("time")
        .pipe(fill_gaps, "time")
        .with_columns(
            lookback_start=pl.col("time").shift(lookback - 1),
            window_start=pl.col("time").shift(-forecast_lead_time),
            window_stop=pl.col("time").shift(
                -(forecast_lead_time + forecast_window - 1)
            ),
        )
    )
    capacity = (
        windpower.select(pl.col(f"ELSPOT NO{k}").max() for k in range(1, 5))
        .collect()
        .to_numpy()[0]
    )
    windpower = (
        windpower.with_columns(
            pl.col(f"ELSPOT NO{k}").forward_fill() / capacity[k - 1]
            for k in range(1, 5)
        )
        .drop_nulls()
        .collect()
    )

    times = (
        windpower.filter(pl.col("time").dt.hour() == 9)
        .select("time", "lookback_start", "window_start", "window_stop")
        .drop_nulls()
    )
    bidding_areas = [
        "ELSPOT NO1",
        "ELSPOT NO2",
        "ELSPOT NO3",
        "ELSPOT NO4",
    ]
    N = len(times)
    X = np.zeros((N, 4, lookback), dtype=np.float32)
    y = np.zeros((N, 4, forecast_window), dtype=np.float32)
    for i, (time, lookback_start, window_start, window_stop) in enumerate(
        times.iter_rows()
    ):
        print(time, lookback_start, window_start, window_stop)
        X[i] = (
            windpower.filter(
                pl.col("time") >= lookback_start,
                pl.col("time") <= time,
            )
            .select(bidding_areas)
            .to_numpy()
            .T
        )

        y[i] = (
            windpower.filter(
                pl.col("time") >= window_start,
                pl.col("time") <= window_stop,
            )
            .select(bidding_areas)
            .to_numpy()
            .T
        )

    return times, X, y
