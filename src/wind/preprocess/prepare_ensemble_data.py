from itertools import chain

import numpy as np
import polars as pl

TAU = 2 * np.pi

ENSEMBLE_MEMBERS = list(range(15))

SHARED_FEATURES = [
    "lt",
    "operating_power_max",
    "mean_production",
    "num_turbines",
    "ELSPOT NO1",
    "ELSPOT NO2",
    "ELSPOT NO3",
    "ELSPOT NO4",
    "now_air_temperature_2m",
    "now_air_pressure_at_sea_level",
    "now_relative_humidity_2m",
    "now_precipitation_amount",
    "now_wind_speed_10m",
    "now_wind_direction_10m",
    "now_air_density",
    "location_mean_ws",
    "now_wind_power_density",
    "sin_hod",
    "cos_hod",
    "sin_doy",
    "cos_doy",
]

ENSEMBLE_FEATURES = [
    "ws10m",
    "wd10m",
    "t2m",
    "rh2m",
    "mslp",
    "g10m",
    "wind_alignment",
    "air_density",
    "gust_factor",
    "ws_power_scaled",
    "ws_turbine_scaled",
    "wind_power_density",
    "wind_power_density_scaled",
]


def get_ensemble_member_features(em: int) -> list[str]:
    return [f"{feature_name}_{em:02d}" for feature_name in ENSEMBLE_FEATURES]


def get_all_features_for_ensemble_member(em: int) -> list[str]:
    return [*SHARED_FEATURES, *get_ensemble_member_features(em)]


def get_all_ensemble_features() -> list[str]:
    return chain.from_iterable(
        get_ensemble_member_features(em) for em in ENSEMBLE_MEMBERS
    )


def air_density(em: int) -> pl.Expr:
    return (pl.col(f"mslp_{em:02d}") / (273.15 + pl.col(f"t2m_{em:02d}"))).alias(
        f"air_density_{em:02d}"
    )


def wind_x(em: int) -> pl.Expr:
    return pl.col(f"ws10m_{em:02d}") * (
        TAU * pl.col(f"wd10m_{em:02d}") / 360
    ).cos().alias(f"wind_x_{em:02d}")


def wind_y(em: int) -> pl.Expr:
    return pl.col(f"ws10m_{em:02d}") * (
        TAU * pl.col(f"wd10m_{em:02d}") / 360
    ).sin().alias(f"wind_y_{em:02d}")


def ws_power_scaled(em: int) -> pl.Expr:
    return (pl.col(f"ws10m_{em:02d}") * pl.col("operating_power_max")).alias(
        f"ws_power_scaled_{em:02d}"
    )


def ws_turbine_scaled(em: int) -> pl.Expr:
    return (pl.col(f"ws10m_{em:02d}") * pl.col("num_turbines")).alias(
        f"ws_turbine_scaled_{em:02d}"
    )


def gust_factor(em: int) -> pl.Expr:
    return (pl.col(f"g10m_{em:02d}") / pl.col(f"ws10m_{em:02d}")).alias(
        f"gust_factor_{em:02d}"
    )


def wind_alignment(em: int) -> pl.Expr:
    return (
        (
            pl.col(f"wind_x_{em:02d}")
            .rolling_mean(5, min_samples=1, center=True)
            .over(["time_ref", "windpark_name"], order_by="lt")
            ** 2
            + pl.col(f"wind_y_{em:02d}")
            .rolling_mean(5, min_samples=1, center=True)
            .over(["time_ref", "windpark_name"], order_by="lt")
            ** 2
        )
        .sqrt()
        .alias(f"wind_alignment_{em:02d}")
    )


def wind_power_density(em: int) -> pl.Expr:
    return (
        pl.col(f"air_density_{em:02d}") * pl.col(f"ws10m_{em:02d}").clip(0, 20) ** 3
    ).alias(f"wind_power_density_{em:02d}")


def wind_power_density_scaled(em: int) -> pl.Expr:
    return (
        pl.col(f"wind_power_density_{em:02d}") * pl.col("operating_power_max")
    ).alias(f"wind_power_density_scaled_{em:02d}")


def get_local_windpower(path: str) -> pl.LazyFrame:
    from datetime import date

    local_windpower = pl.scan_csv(
        path,
        separator=";",
        decimal_comma=True,
        infer_schema=False,
    )

    local_windpower_id = (
        local_windpower.slice(0, 1)
        .drop("time")
        .unpivot(variable_name="windpark_nve", value_name="windpark_nve_id")
        .with_columns(windpark_nve_id=pl.col("windpark_nve_id").cast(pl.Int64))
    )

    local_windpower = (
        local_windpower.slice(1)
        .select(
            # This time is in UTC+1 tz, we convert to UTC, then drop tz since all other datetimes are UTC.
            pl.col("time")
            .str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="ns", time_zone="+01:00")
            .dt.convert_time_zone("UTC")
            .dt.replace_time_zone(None),
            pl.exclude("time").str.replace(",", ".").cast(pl.Float32),
        )
        .filter(pl.col("time") > date(2020, 1, 1))
        .unpivot(index="time", variable_name="windpark_nve", value_name="local_power")
        .drop_nulls()
        .join(local_windpower_id, on="windpark_nve")
    )
    return local_windpower


if __name__ == "__main__":
    weather_forecast_path = "data/met_forecast.parquet"
    weather_nowcast_path = "data/met_nowcast.parquet"
    hour = pl.col("time").dt.hour().cast(pl.Float64)  # 0..23
    doy = pl.col("time").dt.ordinal_day().cast(pl.Float64)  # 1..365/366
    doy_frac = (doy - 1 + hour / 24.0) / 365.2425  # ~[0,1)

    weather_forecast = pl.scan_parquet(weather_forecast_path).with_columns(
        *(wind_x(em) for em in ENSEMBLE_MEMBERS),
        *(wind_y(em) for em in ENSEMBLE_MEMBERS),
    )
    weather_nowcast = (
        pl.scan_parquet("data/met_nowcast.parquet")
        .select(
            pl.col("windpark").alias("sid"),
            pl.col("time").alias("time_ref"),
            pl.exclude("windpark", "time").name.prefix("now_"),
            # pl.col('air_temperature_2m').alias("t2m"),
            # pl.col('air_pressure_at_sea_level').alias("mslp"),
            # pl.col('relative_humidity_2m').alias("rh2m"),
            # pl.col('precipitation_amount').alias(""),
            # pl.col('wind_speed_10m').alias("ws10m"),
            # pl.col('wind_direction_10m').alias("wd10m"),
        )
        .with_columns(
            now_wind_x=pl.col("now_wind_speed_10m")
            * (TAU * pl.col("wind_direction_10m") / 360).cos(),
            now_wind_y=pl.col("now_wind_speed_10m")
            * (TAU * pl.col("wind_direction_10m") / 360).sin(),
            now_air_density=pl.col("now_air_pressure_at_sea_level")
            / (273.15 + pl.col("now_air_temperature_2m")),
            # location_mean_ws=(
            #     pl.col("now_wind_speed_10m").cum_sum()
            #     / pl.col("now_wind_speed_10m").cum_count()
            # ).over(partition_by="sid", order_by="time_ref"),
            # The commented calculation is more correct to avoid information leakage,
            # but this is much easier to compute
            location_mean_ws=pl.col("now_wind_speed_10m")
            .mean()
            .over(partition_by="sid"),
        )
        .with_columns(
            ewm_now_wind_x=pl.col("now_wind_x")
            .ewm_mean("time", alpha=0.5)
            .over("sid", order_by="time"),
            ewm_now_wind_y=pl.col("now_wind_y")
            .ewm_mean("time", alpha=0.5)
            .over("sid", order_by="time"),
        )
        .with_columns(
            now_wind_power_density=pl.col("now_air_density")
            * pl.col("now_wind_speed_10m").clip(0, 20) ** 3
        )
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

    local_windpower = get_local_windpower("data/windpower2002-2024_utcplus1.csv")

    windparks = pl.scan_csv(
        "data/windparks_enriched.csv", try_parse_dates=True
    ).with_columns(
        (pl.col("bidding_area") == f"ELSPOT NO{k}").alias(f"ELSPOT NO{k}")
        for k in range(1, 5)
    )

    data = (
        weather_forecast.join(windparks, left_on="sid", right_on="substation_name")
        .filter(
            pl.col("time_ref") > pl.col("prod_start_new"),
            # pl.col("time_ref") >= date(2021, 1, 1),
        )
        .join(windpower, on=["bidding_area", "time"], how="left")
        .join(local_windpower, on=["time", "windpark_nve_id"], how="left")
        .join(weather_nowcast, on=["sid", "time_ref"])
        .with_columns(
            (TAU * hour / 24.0).sin().alias("sin_hod"),
            (TAU * hour / 24.0).cos().alias("cos_hod"),
            (TAU * doy_frac).sin().alias("sin_doy"),
            (TAU * doy_frac).cos().alias("cos_doy"),
        )
        .with_columns(
            *(air_density(em) for em in ENSEMBLE_MEMBERS),
            *(ws_power_scaled(em) for em in ENSEMBLE_MEMBERS),
            *(ws_turbine_scaled(em) for em in ENSEMBLE_MEMBERS),
            *(gust_factor(em) for em in ENSEMBLE_MEMBERS),
            *(wind_alignment(em) for em in ENSEMBLE_MEMBERS),
        )
        .with_columns(*(wind_power_density(em) for em in ENSEMBLE_MEMBERS))
        .with_columns(*(wind_power_density_scaled(em) for em in ENSEMBLE_MEMBERS))
        .select(
            "time_ref",
            "time",
            "sid",
            "windpark_name",
            "bidding_area",
            "power",
            "local_power",
            *SHARED_FEATURES,
            *get_all_ensemble_features(),
        )
        .sort("time_ref", "time", "bidding_area")
    )

    data.sink_parquet("data/windpower_ensemble_dataset.parquet")
