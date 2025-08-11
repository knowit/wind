import numpy as np
import polars as pl
import xarray as xr

weather_forecast = pl.scan_parquet("data/met_forecast.parquet")
weather_nowcast = pl.scan_parquet("data/met_nowcast.parquet")
windpower = pl.scan_parquet("data/wind_power_per_bidzone.parquet").rename(
    {"__index_level_0__": "time"}
)
windparks = pl.scan_csv("data/windparks_bidzone.csv", try_parse_dates=True).filter(
    pl.col("eic_code") == pl.col("eic_code").first().over("substation_name")
)

bid_zone = "ELSPOT NO3"
zarr_path = "data/weather_forecast.zarr"
ensemble_member = 0
X = (
    weather_forecast.join(windparks, left_on="sid", right_on="substation_name")
    .filter(
        pl.col("bidding_area") == bid_zone,
        pl.col("time_ref") > pl.col("prod_start_new"),
    )
    .select(
        "sid",
        # "prod_start_new",
        "time_ref",
        "time",
        "lt",
        "operating_power_max",
        f"ws10m_{ensemble_member:02d}",
        f"t2m_{ensemble_member:02d}",
        f"rh2m_{ensemble_member:02d}",
        f"mslp_{ensemble_member:02d}",
        f"g10m_{ensemble_member:02d}",
    )
)

# Define dimensions
features = [
    "lt",
    "operating_power_max",
    "ws10m_00",
    "t2m_00",
    "rh2m_00",
    "mslp_00",
    "g10m_00",
]

# Unique sorted values for dimensions
times = X.select("time").unique().collect().to_series().sort()
forecast_steps = X.select("lt").unique().collect().to_series().sort()
num_stations = X.select(pl.n_unique("sid")).collect().item()
sids = list(range(num_stations))

# Mapping for fast lookup
times_idx = {t: i for i, t in enumerate(times)}
# forecast_idx = {t: i for i, t in enumerate(forecast_steps)}
num_features = len(features)

# Initialize dense array with NaNs
shape = (len(times_idx), len(forecast_steps), num_stations, num_features)
data = np.full(shape, np.nan, dtype=np.float32)

# Fill values
for (time_ref, forecast_step), group_data in X.collect(engine="streaming").group_by(
    "time", "lt", maintain_order=False
):
    i = times_idx[time_ref]
    h = group_data.height
    print(time_ref, forecast_step)
    data[i, forecast_step, :h, :] = group_data.select(features).to_numpy()

# Create xarray DataArray
da = xr.DataArray(
    data,
    dims=["time", "forecast_step", "park", "feature"],
    coords={
        "time": times,
        "forecast_step": forecast_steps,
        "park": sids,
        "feature": features,
    },
    name="weather_features",
)

# Save to zarr
da.to_dataset().to_zarr(zarr_path, mode="w")
print(f"Saved DataArray to {zarr_path}")
