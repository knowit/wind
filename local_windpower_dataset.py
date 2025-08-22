import numpy as np
import polars as pl

ensemble_member = 0
weather_forecast_path = "data/met_forecast.parquet"
weather_nowcast_path = "data/met_nowcast.parquet"


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


def get_local_windpower(path: str) -> pl.LazyFrame:
    from datetime import date

    windpower_local = pl.scan_csv(
        path,
        separator=";",
        decimal_comma=True,
        infer_schema=False,
    )

    windpower_local_id = (
        windpower_local.slice(0, 1)
        .drop("time")
        .unpivot(variable_name="windpark_nve", value_name="windpark_nve_id")
        .with_columns(windpark_nve_id=pl.col("windpark_nve_id").cast(pl.Int64))
    )

    windpower_local = (
        windpower_local.slice(1)
        .select(
            pl.col("time").str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="ns"),
            pl.exclude("time").str.replace(",", ".").cast(pl.Float32),
        )
        .filter(pl.col("time") > date(2020, 1, 1))
        .unpivot(index="time", variable_name="windpark_nve", value_name="power")
        .drop_nulls()
        .join(windpower_local_id, on="windpark_nve")
    )
    return windpower_local


# Define dimensions
features = [
    "lt",
    "operating_power_max",
    "mean_production",
    "num_turbines",
    # "ELSPOT NO1",
    # "ELSPOT NO2",
    # "ELSPOT NO3",
    # "ELSPOT NO4",
    "wind_power_potential",
    "wind_turbine_potential",
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

windparks = pl.scan_csv("data/windparks_enriched.csv", try_parse_dates=True)
windpower = get_local_windpower("data/windpower2002-2024_utcplus1.csv")


data = (
    weather_forecast.join(windparks, left_on="sid", right_on="substation_name")
    .filter(
        # pl.col("bidding_area") == bid_zone,
        pl.col("time_ref") > pl.col("prod_start_new"),
    )
    .join(weather_nowcast, on=["sid", "time_ref"])
    .drop_nulls("windpark_nve_id")
    .join(windpower, on=["time", "windpark_nve_id"], how="left")
    .with_columns(
        (pl.col("ws10m_00") * pl.col("operating_power_max")).alias(
            "wind_power_potential"
        ),
        (pl.col("ws10m_00") * pl.col("num_turbines")).alias("wind_turbine_potential"),
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
        "time_ref",
        "time",
        "bidding_area",
        "sid",
        "windpark_name",
        "power",
        *features,
    )
    .sort("time_ref", "time", "bidding_area")
)
data.sink_parquet("data/local_windpower.parquet")
