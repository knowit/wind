import numpy as np
import polars as pl
import torch

bid_zone = "ELSPOT NO3"
zarr_path = "data/weather_forecast.zarr"
ensemble_member = 0


TAU = 2 * np.pi  # 2π
hour = pl.col("time").dt.hour().cast(pl.Float64)  # 0..23
doy = pl.col("time").dt.ordinal_day().cast(pl.Float64)  # 1..365/366
weekday = pl.col("time").dt.weekday().cast(pl.Float64)
doy_frac = (doy - 1 + hour / 24.0) / 365.2425  # ~[0,1)
dow_frac = (weekday - 1 + hour / 24.0) / 7

weather_forecast = pl.scan_parquet("data/met_forecast.parquet")
weather_nowcast = pl.scan_parquet("data/met_nowcast.parquet").select(
    pl.col("windpark").alias("sid"),
    "time",
    pl.exclude("windpark", "time").name.prefix("now_"),
)
windpower = (
    pl.scan_parquet("data/wind_power_per_bidzone.parquet")
    .select(
        pl.col("__index_level_0__").alias("time"),
        pl.col(bid_zone).alias("bid_zone_power"),
    )
    .with_columns(issue_date=pl.col("time").dt.date())
)
windpower_prev = windpower.group_by(
    issue_date=pl.col("issue_date") + pl.duration(days=1)
).agg(
    last_day_mean=pl.col("bid_zone_power").mean(),
    last_value=pl.col("bid_zone_power").sort_by("time").last(),
)
windpower = windpower.join(windpower_prev, on="issue_date")

windparks = pl.scan_csv("data/windparks_bidzone.csv", try_parse_dates=True)


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
    "lt",
    "operating_power_max",
    "last_day_mean",
    "last_value",
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

data = (
    weather_forecast.join(windparks, left_on="sid", right_on="substation_name")
    .filter(
        pl.col("bidding_area") == bid_zone,
        pl.col("time_ref") > pl.col("prod_start_new"),
    )
    .join(windpower, on="time", how="inner")
    .join(weather_nowcast, on=["sid", "time"])
    .with_columns(
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
        # "prod_start_new",
        "time_ref",
        "time",
        *features,
    )
    .sort("time_ref", "time")
)

# Unique sorted values for dimensions
times = data.select("time_ref", "time").unique(maintain_order=True).collect()
num_stations = data.select(pl.n_unique("sid")).collect().item()
sids = list(range(num_stations))

num_features = len(features)


# Initialize dense array with NaNs
shape = (times.height, num_stations, num_features)
# X = np.full(shape, np.nan, dtype=np.float32)
X = np.zeros(shape, dtype=np.float32)

# Fill values
for i, partition in enumerate(
    data.collect(engine="streaming").partition_by(
        "time_ref", "time", maintain_order=True
    )
):
    h = partition.height
    print(i, h)
    X[i, :h, :] = partition.select(features).to_numpy()


y = (
    times.join(windpower.collect(), on="time", how="left")
    .select("bid_zone_power")
    .to_numpy()[:, 0]
)
print(X.shape, y.shape)

torch.save(
    {
        "X": torch.from_numpy(X),  # or convert later on load
        "y": torch.from_numpy(y),
    },
    "data/torch_dataset.pt",
)
