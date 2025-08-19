import cf_xarray as cfxr
import numpy as np
import pandas as pd
import polars as pl
import torch
import xarray as xr

ensemble_member = 0
weather_forecast_path = "data/met_forecast.parquet"
weather_nowcast_path = "data/met_nowcast.parquet"

TAU = 2 * np.pi
hour = pl.col("time").dt.hour().cast(pl.Float64)  # 0..23
doy = pl.col("time").dt.ordinal_day().cast(pl.Float64)  # 1..365/366
weekday = pl.col("time").dt.weekday().cast(pl.Float64)
doy_frac = (doy - 1 + hour / 24.0) / 365.2425  # ~[0,1)
dow_frac = (weekday - 1 + hour / 24.0) / 7

weather_forecast = pl.scan_parquet(weather_forecast_path)
weather_nowcast = pl.scan_parquet("data/met_nowcast.parquet").select(
    pl.col("windpark").alias("sid"),
    pl.col("time").alias("time_ref"),
    pl.exclude("windpark", "time").name.prefix("now_"),
)
windpower = (
    (
        pl.scan_parquet("data/wind_power_per_bidzone.parquet").rename(
            {"__index_level_0__": "time"}
        )
    )
    .unpivot(index="time", variable_name="bidding_area", value_name="power")
    .with_columns(
        (pl.col("bidding_area") == f"ELSPOT NO{k}").alias(f"ELSPOT NO{k}")
        for k in range(1, 5)
    )
)

windparks = pl.scan_csv("data/windparks_enriched.csv", try_parse_dates=True)


def ensemble_mean(variable):
    return pl.mean_horizontal(pl.col(f"{variable}_{k:02d}" for k in range(15))).alias(
        f"{variable}_mean"
    )


def ensemble_std(variable):
    return (
        pl.concat_list(pl.col(f"{variable}_{k:02d}" for k in range(15)))
        .list.std()
        .alias(f"{variable}_std")
    )


# Define dimensions
features = [
    "not_missing",
    "lt",
    "operating_power_max",
    "mean_production",
    "num_turbines",
    "ELSPOT NO1",
    "ELSPOT NO2",
    "ELSPOT NO3",
    "ELSPOT NO4",
    "last_day_mean",
    "last_day_std",
    "last_day_min",
    "last_day_max",
    "last_value",
    "last_values_mean",
    "ws10m_00",
    "wd10m_00",
    "t2m_00",
    "rh2m_00",
    "mslp_00",
    "g10m_00",
    "ws10m_mean",
    "t2m_mean",
    "rh2m_mean",
    "mslp_mean",
    "g10m_mean",
    "ws10m_std",
    "t2m_std",
    "rh2m_std",
    "mslp_std",
    "g10m_std",
    "now_air_temperature_2m",
    "now_air_pressure_at_sea_level",
    "now_relative_humidity_2m",
    "now_precipitation_amount",
    "now_wind_speed_10m",
    "now_wind_direction_10m",
    "sin_hod",
    "cos_hod",
    "sin_doy",
    "cos_doy",
    "sin_dow",
    "cos_dow",
]

data_joined = (
    weather_forecast.join(windparks, left_on="sid", right_on="substation_name")
    .filter(
        # pl.col("bidding_area") == bid_zone,
        pl.col("time_ref") > pl.col("prod_start_new"),
    )
    .join(windpower, on=["bidding_area", "time"], how="left")
    .join(weather_nowcast, on=["sid", "time_ref"])
)
windpower_features = (
    data_joined.select("bidding_area", "time_ref")
    .unique()
    .join(windpower, on="bidding_area")
    .filter(
        pl.col("time") < pl.col("time_ref"),
        pl.col("time") >= pl.col("time_ref") - pl.duration(hours=24),
    )
    .group_by("bidding_area", "time_ref")
    .agg(
        last_day_mean=pl.col("power").mean(),
        last_day_std=pl.col("power").std(),
        last_day_min=pl.col("power").min(),
        last_day_max=pl.col("power").max(),
        last_value=pl.col("power").sort_by("time").last(),
        last_values_mean=pl.col("power")
        .filter(pl.col("time") >= pl.col("time_ref") - pl.duration(hours=3))
        .mean(),
    )
)

data = (
    data_joined.join(windpower_features, on=["bidding_area", "time_ref"], how="left")
    .with_columns(
        pl.lit(1).alias("not_missing"),
        ensemble_mean("ws10m"),
        ensemble_mean("t2m"),
        ensemble_mean("rh2m"),
        ensemble_mean("mslp"),
        ensemble_mean("g10m"),
        ensemble_std("ws10m"),
        ensemble_std("t2m"),
        ensemble_std("rh2m"),
        ensemble_std("mslp"),
        ensemble_std("g10m"),
        # Hour-of-day (period = 24)
        (TAU * hour / 24.0).sin().alias("sin_hod"),
        (TAU * hour / 24.0).cos().alias("cos_hod"),
        # Day-of-year (period ≈ 365.2425 to handle leap years smoothly)
        (TAU * doy_frac).sin().alias("sin_doy"),
        (TAU * doy_frac).cos().alias("cos_doy"),
        (TAU * dow_frac).sin().alias("sin_dow"),
        (TAU * dow_frac).cos().alias("cos_dow"),
    )
    .select(
        "sid",
        "windpark_name",
        # "prod_start_new",
        "bidding_area",
        "time_ref",
        "time",
        *features,
    )
    .sort("bidding_area", "time_ref", "time")
)

# Unique sorted values for dimensions
forecast_index = (
    data.select("time_ref", "time", "lt", "bidding_area")
    .unique(maintain_order=True)
    .collect()
)
# num_stations = data.select(pl.col("sid").n_unique()).collect().item()
num_stations = (
    data.group_by("bidding_area")
    .agg(n=pl.col("windpark_name").n_unique())
    .select(pl.col("n").max())
    .collect()
    .item()
)

num_features = len(features)


# Initialize dense array with NaNs
shape = (forecast_index.height, num_stations, num_features)
# X = np.full(shape, np.nan, dtype=np.float32)
X = np.zeros(shape, dtype=np.float32)

# Fill values
for i, partition in enumerate(
    data.collect(engine="streaming").partition_by(
        "time_ref", "time", "bidding_area", maintain_order=True
    )
):
    h = partition.height
    print(i, h)
    X[i, :h, :] = partition.select(features).to_numpy()


y = (
    forecast_index.join(windpower.collect(), on=["bidding_area", "time"], how="left")
    .select("power")
    .to_numpy()[:, 0]
    .astype(np.float32)
)
X_missing_idx = np.any((np.isnan(X)), axis=(1, 2))
y_missing_idx = np.isnan(y)
not_missing_idx = ~X_missing_idx & ~y_missing_idx
print("X missing:", np.sum(X_missing_idx))
print("y missing:", np.sum(y_missing_idx))
print("No missing target:", X[not_missing_idx].shape, y[not_missing_idx].shape)

torch.save(
    {
        "X": torch.from_numpy(X[not_missing_idx]),
        "y": torch.from_numpy(y[not_missing_idx]),
    },
    "data/torch_dataset_all_zones.pt",
)


da_forecast_index = pd.MultiIndex.from_frame(
    forecast_index.to_pandas(), names=["time_ref", "time", "lt", "bidding_area"]
)

da_X = xr.DataArray(
    X,
    dims=("forecast_index", "station", "feature"),
    coords={
        "forecast_index": da_forecast_index,
        "station": np.array(range(num_stations)),
        "feature": features,
    },
    name="X",
)

da_y = xr.DataArray(
    y,
    dims=("forecast_index",),
    coords={
        "forecast_index": da_forecast_index,
    },
    name="y",
)

ds = xr.Dataset(dict(X=da_X, y=da_y))
ds_encoded = cfxr.encode_multi_index_as_compress(ds, "forecast_index")
ds_encoded.to_zarr("data/dataset_all_zones.zarr", mode="w")
