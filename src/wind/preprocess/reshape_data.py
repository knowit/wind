from datetime import datetime

import cf_xarray as cfxr
import numpy as np
import pandas as pd
import polars as pl
import torch
import xarray as xr

from prepare_data import get_all_features_for_ensemble_member


def add_lagged_windpower(data, windpower):
    lagged_windpower = (
        data.select("time_ref", "bidding_area")
        .unique()
        .join(windpower, on="bidding_area")
        .filter(
            pl.col("time") < pl.col("time_ref"),
            pl.col("time") >= pl.col("time_ref") - pl.duration(hours=24),
        )
        .group_by("time_ref", "bidding_area")
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
    return data.join(lagged_windpower, on=["time_ref", "bidding_area"], how="left")


windpower = (
    pl.scan_parquet("data/wind_power_per_bidzone.parquet").rename(
        {"__index_level_0__": "time"}
    )
).unpivot(index="time", variable_name="bidding_area", value_name="power")


em = 0
local_preds = pl.scan_parquet("data/local_power_pred.parquet").select(
    "time_ref",
    "time",
    "sid",
    "windpark_name",
    local_power_pred=pl.col(f"local_power_pred_{em:02d}").clip(0, None),
)

data = (
    # pl.scan_parquet("data/windpower_dataset.parquet")
    pl.scan_parquet("data/windpower_ensemble_dataset.parquet")
    .join(local_preds, on=["time_ref", "time", "windpark_name"])
    .with_columns(mask=pl.lit(True))
    .sort("time_ref", "time", "bidding_area")
)


# Unique sorted values for dimensions
forecast_index = (
    data.select("time_ref", "time", "lt", "bidding_area")
    .unique(maintain_order=True)
    .collect()
)

num_stations = (
    data.group_by("bidding_area")
    .agg(n=pl.col("windpark_name").n_unique())
    .select(pl.col("n").max())
    .collect()
    .item()
)

# Lagged power features are not available to the local model and are only used for the bidding area model
lagged_power_features = [
    "last_day_mean",
    "last_day_std",
    "last_day_min",
    "last_day_max",
    "last_value",
    "last_values_mean",
]
features = [*get_all_features_for_ensemble_member(em), "local_power_pred", "mask"]
num_features = len(features)


# Initialize dense array with NaNs
shape = (forecast_index.height, num_stations, num_features)
# X = np.full(shape, np.nan, dtype=np.float32)
X = np.zeros(shape, dtype=np.float32)
print(shape)
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
