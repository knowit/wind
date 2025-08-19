# --- libraries ---
from typing import Tuple

import numpy as np
import pandas as pd
import polars as pl
from statsforecast.core import StatsForecast
from statsforecast.models import MSTL  # multi-seasonal STL with fast ETS/ARIMA trend
from statsmodels.tsa.statespace.sarimax import SARIMAX


# --- helpers ---
def fill_time_gaps(windpower: pl.DataFrame) -> pl.DataFrame:
    start_time = windpower.select(pl.col("time").min()).item()
    end_time = windpower.select(pl.col("time").max()).item()
    time_range = pl.datetime_range(
        start_time, end_time, interval="1h", time_unit="ns", eager=True
    ).to_frame("time")
    windpower_filled = time_range.join(windpower, on="time", how="left").with_columns(
        power=pl.col("power").forward_fill()
    )
    assert windpower.select(pl.col("power").is_null().sum()).item() == 0
    return windpower_filled


def _to_pandas_hourly(windpower_pl: pl.DataFrame) -> pd.Series:
    """Convert Polars -> pandas, ensure hourly continuity, avoid leakage in fills."""
    df = windpower_pl.select(
        pl.col("time").cast(pl.Datetime), pl.col("power").cast(pl.Float64)
    ).sort("time")

    s = df.to_pandas()
    s["time"] = pd.to_datetime(s["time"], utc=False)
    s = s.drop_duplicates("time").set_index("time").sort_index()
    # Ensure hourly index; forward-fill only (no look-ahead leakage)
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="h")
    y = s.reindex(full_idx)["power"].ffill()
    # If the very beginning is NaN, backfill just the initial stretch
    if y.isna().any():
        y = y.bfill()
    y.name = "power"
    return y


def _fourier(
    index: pd.DatetimeIndex, period_hours: float, K: int, prefix: str
) -> pd.DataFrame:
    """
    Calendar Fourier series with hourly period.
    Uses absolute hour-count from epoch so features are time-invariant to window.
    """
    # hours since epoch (float is fine)
    t_hours = (index.view("int64") // 10**9) / 3600.0
    cols = {}
    for k in range(1, K + 1):
        ang = 2.0 * np.pi * k * t_hours / period_hours
        cols[f"{prefix}_sin{k}"] = np.sin(ang)
        cols[f"{prefix}_cos{k}"] = np.cos(ang)
    X = pd.DataFrame(cols, index=index)
    return X


def _fit_and_forecast(
    y: pd.Series,
    horizon: int,
    K_yearly: int = 6,
    sarima_order=(1, 0, 1),
    sarima_seasonal=(1, 1, 1, 24),
    last_params: np.ndarray | None = None,
):
    """Fit SARIMAX with 24h seasonality + yearly Fourier exog; return (yhat, conf_int, params)."""
    # yearly period in hours (approx leap-aware)
    YEAR_HOURS = 24.0 * 365.25

    # exog for train and for future horizon
    # X_train = _fourier(y.index, period_hours=YEAR_HOURS, K=K_yearly, prefix="yr")
    # future_index = pd.date_range(
    #     y.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h"
    # )
    # X_future = _fourier(future_index, period_hours=YEAR_HOURS, K=K_yearly, prefix="yr")

    mod = SARIMAX(
        y,
        order=sarima_order,
        seasonal_order=sarima_seasonal,  # 24h seasonality
        # exog=X_train,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fit_kwargs = dict(disp=False, maxiter=10)
    if last_params is not None:
        # warm start for speed across rolling cutoffs
        fit_kwargs["start_params"] = last_params
        fit_kwargs["maxiter"] = 500

    res = mod.fit(**fit_kwargs)
    fc = res.get_forecast(steps=horizon)
    yhat = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    return yhat, ci, res.params


def build_ts_feature_forecasts(
    windpower: pl.DataFrame,
    time_refs: pl.DataFrame,
    horizon: int = 48,
    K_yearly: int = 6,
    sarima_order=(1, 0, 1),
    sarima_seasonal=(1, 1, 1, 24),
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    For each time_ref in weather_forecast, fit on data up to that point and forecast next `horizon` hours.

    Returns:
      - ts_long:  [time_ref, horizon_h, time, power_forecast, lo, hi]
      - ts_wide:  [time_ref, power_fcst_h1, ..., power_fcst_h{H}]
      - weather_with_features: weather_forecast joined with ts_wide on time_ref
    """
    # 1) Prepare series
    y_full = fill_time_gaps(windpower).to_pandas().set_index("time").asfreq("h")

    # 2) Extract sorted unique time_ref values (as pandas Timestamps)
    trefs = time_refs.to_pandas()["time_ref"]
    # trefs = pd.to_datetime(trefs)

    # Keep only those within (min(y_full), max(y_full)] so we have history
    trefs = trefs[(trefs > y_full.index.min()) & (trefs <= y_full.index.max())]
    if len(trefs) == 0:
        raise ValueError("No usable time_ref values fall within the windpower history.")

    horizon_h = np.arange(1, horizon + 1, dtype=int)
    rows_long = []
    last_params = None

    for tr in trefs:
        # Train on all data up to tr (inclusive)
        y = y_full.loc[:tr]
        print(tr, y.shape)
        # Fit & forecast
        yhat, ci, last_params = _fit_and_forecast(
            y=y,
            horizon=horizon,
            K_yearly=K_yearly,
            sarima_order=sarima_order,
            sarima_seasonal=sarima_seasonal,
            last_params=last_params,  # warm start for speed across rolling refs
        )
        df_fc = pd.DataFrame(
            {
                "time_ref": tr,
                "horizon_h": horizon_h,
                "time": yhat.index,
                "power_forecast": yhat.values,
                "power_forecast_lo": ci["lower power"],
                "power_forecast_hi": ci["upper power"],
            }
        )
        # df_fc["lo"] = ci["yhat_lo"].values
        # df_fc["hi"] = ci["yhat_hi"].values
        rows_long.append(df_fc)

    ts_long_pd = pd.concat(rows_long, ignore_index=True)
    ts_long = pl.from_pandas(ts_long_pd)

    # # 3) Wide format (one row per time_ref, columns h=1..H)
    # ts_wide = (
    #     ts_long.select(["time_ref", "horizon_h", "power_forecast"])
    #     .pivot(values="power_forecast", index="time_ref", columns="horizon_h")
    #     .sort("time_ref")
    # )
    # # Rename h columns -> power_fcst_h{h}
    # rename_map = {
    #     c: (
    #         f"power_fcst_h{c}"
    #         if isinstance(c, (int, np.integer)) or str(c).isdigit()
    #         else c
    #     )
    #     for c in ts_wide.columns
    #     if c != "time_ref"
    # }
    # ts_wide = ts_wide.rename(rename_map)

    # # 4) Join onto your weather_forecast rows
    # weather_with_features = time_refs.join(ts_wide, on="time_ref", how="left")

    return ts_long  # , ts_wide, weather_with_features


def build_mstl_feature_forecasts(
    windpower: pl.DataFrame,
    time_refs: pl.DataFrame,
    horizon: int = 48,
):
    # 1) Polars -> pandas in StatsForecast format
    df = (
        windpower.select(
            pl.col("time").cast(pl.Datetime).alias("ds"),
            pl.col("power").cast(pl.Float64).alias("y"),
        )
        .sort("ds")
        .to_pandas()
    )
    df["unique_id"] = "wp"

    # 2) Model: 24h + yearly (8760h) seasonality
    # Tip: if memory is tight, drop yearly or use [24, 168] (daily + weekly) — still very good for 48h.
    models = [MSTL(season_length=[24, 8760])]

    sf = StatsForecast(models=models, freq="H", n_jobs=-1, verbose=True)

    # 3) Rolling-origin CV for ALL cutoffs (step_size=1h)
    # Produces forecasts for ds > cutoff, for each cutoff.
    fcv = sf.cross_validation(df=df, h=horizon, step_size=1)

    # Depending on your statsforecast version, the forecast column may be named after the model
    # (e.g., 'MSTL') or 'y_hat'. Handle both:
    yhat_col = (
        "MSTL"
        if "MSTL" in fcv.columns
        else ("y_hat" if "y_hat" in fcv.columns else None)
    )
    if yhat_col is None:
        raise RuntimeError(
            f"Can't find forecast column in cross_validation output: {fcv.columns.tolist()}"
        )

    # 4) Filter to your exact time_ref cutoffs and reshape to long/wide
    wf = time_refs.to_pandas()

    # Keep only the cutoffs you care about
    fcv = fcv[fcv["cutoff"].isin(pd.to_datetime(wf["time_ref"]))].copy()

    # Horizon number = (ds - cutoff) in hours
    fcv["horizon_h"] = (
        (fcv["ds"] - fcv["cutoff"]).dt.total_seconds().div(3600).astype(int)
    )
    fcv = fcv[(fcv["horizon_h"] >= 1) & (fcv["horizon_h"] <= horizon)]

    ts_long_pd = fcv.rename(
        columns={"cutoff": "time_ref", "ds": "time", yhat_col: "power_forecast"}
    )[["time_ref", "horizon_h", "time", "power_forecast"]].sort_values(
        ["time_ref", "horizon_h"]
    )

    ts_long = pl.from_pandas(ts_long_pd)

    # # Wide (power_fcst_h1..hH)
    # ts_wide = (
    #     ts_long.select(["time_ref", "horizon_h", "power_forecast"])
    #     .pivot(values="power_forecast", index="time_ref", columns="horizon_h")
    #     .sort("time_ref")
    #     .rename(
    #         {
    #             c: (
    #                 f"power_fcst_h{c}"
    #                 if isinstance(c, (int, np.integer)) or str(c).isdigit()
    #                 else c
    #             )
    #             for c in ts_long.columns
    #             if c not in {"time_ref", "horizon_h", "power_forecast"}
    #         }
    #     )
    # )

    # weather_with_features = pl.from_pandas(wf).join(ts_wide, on="time_ref", how="left")
    return ts_long  # , ts_wide, weather_with_features


if __name__ == "__main__":
    bidding_area = "ELSPOT NO3"
    windpower = (
        pl.scan_parquet("data/wind_power_per_bidzone.parquet")
        .select(
            pl.col("__index_level_0__").alias("time"),
            pl.col(bidding_area).alias("power"),
        )
        .sort("time")
        .collect()
    )
    time_refs = (
        pl.scan_parquet("data/met_forecast.parquet")
        .select(pl.col("time_ref").unique())
        .sort("time_ref")
        .tail(1000)
        .collect()
    )
    # ts_long = build_ts_feature_forecasts(
    #     windpower=windpower,
    #     time_refs=time_refs,
    #     horizon=48,
    #     K_yearly=6,  # increase to 8-10 if you need richer yearly seasonality
    #     sarima_order=(1, 0, 1),
    #     sarima_seasonal=(
    #         1,
    #         1,
    #         1,
    #         24,
    #     ),  # daily seasonality (24h) with seasonal differencing
    # )

    ts_long = build_mstl_feature_forecasts(
        windpower,
        time_refs,
    )

    # Now use `ts_wide` columns (power_fcst_h1..h48) as input features to your model.
    ts_long.write_csv("data/ts_forecast.csv")
