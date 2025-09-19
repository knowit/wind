from datetime import date

import numpy as np
import polars as pl
import requests

ENSEMBLE_MEMBERS = list(range(15))

LOCAL_FEATURES = [
    "lt",
    "mean_hub_height",
    "mean_rotor_diameter",
    "unavailable_capacity",
    "elevation",
    "ruggedness",
    "ELSPOT NO1",
    "ELSPOT NO2",
    "ELSPOT NO3",
    "ELSPOT NO4",
    "sin_hod",
    "cos_hod",
    "sin_doy",
    "cos_doy",
    "location_mean_ws",
    "now_air_temperature_2m",
    "now_air_pressure_at_sea_level",
    "now_relative_humidity_2m",
    "now_precipitation_amount",
    "now_wind_speed_10m",
    "now_wind_direction_10m",
    "now_air_density",
    "now_wind_alignment",
    "now_wind_power_density",
    "ws10m",
    "wd10m",
    "t2m",
    "rh2m",
    "mslp",
    "g10m",
    "wshh",
    "adhh",
    "phh",
    "wind_power_density",
    "wind_alignment",
    "gust_factor",
]

TAU = 2 * np.pi


def wind_speed_at_hub_height() -> pl.Expr:
    """Wind speed at hub height"""
    alpha = 0.143  # Empirically derived coefficient for estimating wind speed at different heights
    # See: https://en.wikipedia.org/wiki/Wind_profile_power_law
    return (pl.col("ws10m") * ((pl.col("mean_hub_height") / 10) ** alpha)).alias("wshh")


def hub_height_variables():
    # constants
    g = 9.80665  # m/s^2
    Rd = 287.05  # J/(kg*K)
    eps = 0.622

    wshh = wind_speed_at_hub_height()

    # optional lapse adjustment to hub height temperature
    T_layer = pl.col("t2m") + 273.15 - 0.0065 * pl.col("mean_hub_height")

    # 2) vapor pressure from RH (Magnus-Bolton)
    es_hPa = 6.112 * ((17.67 * pl.col("t2m")) / (pl.col("t2m") + 243.5)).exp()
    e = (pl.col("rh2m") / 100.0) * es_hPa * 100.0  # Pa

    # 3) preliminary station pressure at ground (dry)
    pmsl = pl.col("mslp") * 100.0  # Pa

    p0_dry = pmsl * (-g * pl.col("elevation") / (Rd * T_layer)).exp()

    # specific humidity using p0_dry
    q = eps * e / (p0_dry - (1 - eps) * e)
    Tv = T_layer * (1 + 0.61 * q)

    # 4) pressure at hub height
    phh = (
        pmsl
        * (-g * (pl.col("elevation") + pl.col("mean_hub_height")) / (Rd * Tv)).exp()
    )

    # if iterate:
    #     # one quick iteration
    #     q = eps * e / (phh - (1 - eps) * e)
    #     Tv = T_layer * (1 + 0.61*q)
    #     phh = pmsl * np.exp(-g * (H + z) / (Rd * Tv))

    # 5) density
    rho_hh = phh / (Rd * Tv)
    wpd = 0.5 * rho_hh * (wshh.clip(0, 25) ** 3)
    return (
        wshh.alias("wshh"),
        rho_hh.alias("adhh"),
        phh.alias("phh"),
        wpd.alias("wind_power_density"),
    )


def air_density() -> pl.Expr:
    return (pl.col("mslp") / (273.15 + pl.col("t2m"))).alias("air_density")


def wind_power_density(wind_col="ws10m") -> pl.Expr:
    return (pl.col("air_density") * pl.col(wind_col).clip(0, 20) ** 3).alias(
        "wind_power_density"
    )


def wind_x(wind_col="ws10m") -> pl.Expr:
    return (pl.col(wind_col) * (TAU * pl.col("wd10m") / 360).cos()).alias("wind_x")


def wind_y(wind_col="ws10m") -> pl.Expr:
    return (pl.col(wind_col) * (TAU * pl.col("wd10m") / 360).sin()).alias("wind_y")


def gust_factor() -> pl.Expr:
    return (pl.col("g10m") / pl.col("ws10m")).alias("gust_factor")


def wind_alignment(wind_col="ws10m") -> pl.Expr:
    wind_x_comp = wind_x(wind_col)
    wind_y_comp = wind_y(wind_col)
    return (
        (
            wind_x_comp.rolling_mean(3, min_samples=1, center=True).over(
                ["time_ref", "windpark_nve"], order_by="lt"
            )
            ** 2
            + wind_y_comp.rolling_mean(3, min_samples=1, center=True).over(
                ["time_ref", "windpark_nve"], order_by="lt"
            )
            ** 2
        )
        .sqrt()
        .alias("wind_alignment")
    )


def seasonal_features(time_col: str) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
    hour_frac = pl.col(time_col).dt.hour().cast(pl.Float64) / 24.0
    doy = pl.col(time_col).dt.ordinal_day().cast(pl.Float64)
    doy_frac = (doy - 1 + hour_frac) / 365.2425
    return (
        (TAU * hour_frac).sin().alias("sin_hod"),
        (TAU * hour_frac).cos().alias("cos_hod"),
        (TAU * doy_frac).sin().alias("sin_doy"),
        (TAU * doy_frac).cos().alias("cos_doy"),
    )


def bidding_area_dummies() -> list[pl.Expr]:
    return [
        (pl.col("bidding_area") == f"ELSPOT NO{k}").alias(f"ELSPOT NO{k}")
        for k in range(1, 5)
    ]


def get_local_windpower(path: str) -> pl.LazyFrame:
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


def get_weather_forecast(path: str, evaluation_path: str | None = None) -> pl.LazyFrame:
    data = pl.scan_parquet(path)
    if evaluation_path is not None:
        evaluation_data = pl.scan_csv(
            evaluation_path, schema=data.collect_schema(), null_values="NA"
        )
        # for c1, c2 in zip(data.collect_schema(), evaluation_data.collect_schema()):
        #     if c1 != c2:
        #         print("--->", end="")
        #     print(c1, c2)
        data = pl.concat([data, evaluation_data])

    weather_forecast_ensemble_data = []
    for em in ENSEMBLE_MEMBERS:
        em_weather_forecast = data.select(
            pl.col("sid").alias("windpark_statnet"),
            "time_ref",
            "time",
            "lt",
            pl.lit(em).alias("em"),
            pl.col(f"ws10m_{em:02d}").alias("ws10m"),
            pl.col(f"wd10m_{em:02d}").alias("wd10m"),
            pl.col(f"t2m_{em:02d}").alias("t2m"),
            pl.col(f"rh2m_{em:02d}").alias("rh2m"),
            pl.col(f"mslp_{em:02d}").alias("mslp"),
            pl.col(f"g10m_{em:02d}").alias("g10m"),
        )
        weather_forecast_ensemble_data.append(em_weather_forecast)

    weather_forecast = pl.concat(weather_forecast_ensemble_data)
    return weather_forecast


def get_weather_nowcast(path: str, evaluation_path: str | None = None) -> pl.LazyFrame:
    data = pl.scan_parquet(path)
    if evaluation_path is not None:
        evaluation_data = pl.scan_parquet(evaluation_path)
        # for c1, c2 in zip(data.collect_schema(), evaluation_data.collect_schema()):
        #     if c1 != c2:
        #         print("---> ", end="")
        #     print(c1, c2)
        data = pl.concat([data, evaluation_data])
    weather_nowcast = (
        data.select(
            pl.col("windpark").alias("windpark_statnet"),
            pl.col("time").alias("time_ref"),
            pl.exclude("windpark", "time").name.prefix("now_"),
        )
        .with_columns(
            now_wind_x=pl.col("now_wind_speed_10m")
            * (TAU * pl.col("now_wind_direction_10m") / 360).cos(),
            now_wind_y=pl.col("now_wind_speed_10m")
            * (TAU * pl.col("now_wind_direction_10m") / 360).sin(),
            now_air_density=pl.col("now_air_pressure_at_sea_level")
            / (273.15 + pl.col("now_air_temperature_2m")),
            # location_mean_ws=(
            #     pl.col("now_wind_speed_10m").cum_sum()
            #     / pl.col("now_wind_speed_10m").cum_count()
            # ).over(partition_by="windpark_statnet", order_by="time_ref"),
            # The commented calculation is more correct to avoid information leakage,
            # but this is much easier to compute
            location_mean_ws=pl.col("now_wind_speed_10m")
            .mean()
            .over(partition_by="windpark_statnet"),
        )
        .with_columns(
            ewm_now_wind_x=pl.col("now_wind_x")
            .ewm_mean(alpha=0.5)
            .over("windpark_statnet", order_by="time_ref"),
            ewm_now_wind_y=pl.col("now_wind_y")
            .ewm_mean(alpha=0.5)
            .over("windpark_statnet", order_by="time_ref"),
        )
        .with_columns(
            now_wind_alignment=(
                pl.col("ewm_now_wind_x") ** 2 + pl.col("ewm_now_wind_y") ** 2
            ).sqrt(),
            now_wind_power_density=pl.col("now_air_density")
            * pl.col("now_wind_speed_10m").clip(0, 20) ** 3,
        )
    )
    return weather_nowcast


def get_outages() -> pl.DataFrame:
    areas = [
        {"name": "NO1", "code": "10YNO-1--------2"},
        {"name": "NO2", "code": "10YNO-2--------T"},
        {"name": "NO3", "code": "10YNO-3--------J"},
        {"name": "NO4", "code": "10YNO-4--------9"},
    ]
    messages = []
    skip = 0
    while True:
        res = requests.get(
            "https://ummapi.nordpoolgroup.com/messages",
            params={
                "limit": 2000,
                "messageTypes": "ProductionUnavailability",
                "areas": [a["code"] for a in areas],
                "fuelTypes": 19,
                "skip": skip,
            },
        )
        if res.status_code != 200:
            print(res.status_code)
            break

        content = res.json()
        if len(content["items"]) == 0:
            break
        messages.extend(content["items"])
        skip += len(content["items"])
        if skip >= content["total"]:
            break

    production = (
        pl.json_normalize(messages, infer_schema_length=1000)
        .filter(pl.col("messageType") == 1, pl.col("generationUnits").is_null())
        .explode("productionUnits")
        .select(
            pl.col("publicationDate").cast(pl.Datetime),
            pl.col("productionUnits").struct.field("eic").alias("eic"),
            pl.col("productionUnits")
            .struct.field("installedCapacity")
            .alias("installedCapacity"),
            pl.col("productionUnits").struct.field("timePeriods").alias("timePeriods"),
        )
        .explode("timePeriods")
        .with_columns(
            pl.col("timePeriods")
            .struct.field("unavailableCapacity")
            .alias("unavailableCapacity"),
            pl.col("timePeriods")
            .struct.field("availableCapacity")
            .alias("availableCapacity"),
            pl.col("timePeriods")
            .struct.field("eventStart")
            .cast(pl.Datetime)
            .alias("eventStart"),
            pl.col("timePeriods")
            .struct.field("eventStop")
            .cast(pl.Datetime)
            .alias("eventStop"),
        )
    )
    return production


def add_outages(df: pl.LazyFrame) -> pl.LazyFrame:
    eic_lookup = (
        pl.scan_csv("data/windparks_lookup.csv")
        .select("windpark_nve_id", "eic_code")
        .unique()
    )
    outages = get_outages().lazy().join(eic_lookup, left_on="eic", right_on="eic_code")
    windparks_times = df.select("time_ref", "time", "windpark_nve_id").unique()
    outage_periods = (
        windparks_times.join(outages, on="windpark_nve_id")
        .filter(
            pl.col("time_ref") > pl.col("publicationDate"),
            pl.col("time") >= pl.col("eventStart"),
            pl.col("time") <= pl.col("eventStop"),
        )
        .group_by("time_ref", "time", "windpark_nve_id")
        .agg(
            pl.col("unavailableCapacity").sum(),
            pl.col("installedCapacity").sum(),
            pl.col("availableCapacity").sum(),
        )
        .with_columns(
            unavailable_capacity=pl.col("unavailableCapacity")
            / pl.col("installedCapacity")
        )
    )
    return df.join(
        outage_periods, on=["time_ref", "time", "windpark_nve_id"], how="left"
    ).with_columns(unavailable_capacity=pl.col("unavailable_capacity").fill_null(0.0))


def main():
    weather_forecast = get_weather_forecast(
        "data/met_forecast.parquet"  # , evaluation_path="data/met_forecast_daily/*"
    )
    weather_nowcast = get_weather_nowcast(
        "data/met_nowcast.parquet"  # , evaluation_path="data/met_nowcast_daily/*"
    )

    local_windpower = get_local_windpower("data/windpower2002-2024_utcplus1.csv").drop(
        "windpark_nve"
    )
    windparks = pl.scan_csv("data/windparks_enriched.csv", try_parse_dates=True)

    time_index = weather_forecast.select("time_ref", "time").unique()

    data = (
        windparks.join(time_index, how="cross")
        .join(local_windpower, on=["windpark_nve_id", "time"], how="left")
        .join(windparks, on="windpark_nve_id", how="left")
        .join(weather_forecast, on=["windpark_statnet", "time_ref", "time"], how="left")
        .join(weather_nowcast, on=["windpark_statnet", "time_ref"], how="left")
        .filter(
            pl.col("time_ref") > pl.col("production_start_date"),
        )
        .pipe(add_outages)
        .with_columns(
            # wshh(),
            # air_density(),
            gust_factor(),
            *hub_height_variables(),
            *seasonal_features("time"),
            *bidding_area_dummies(),
        )
        .with_columns(
            wind_alignment("wshh"),
            # wind_power_density("wshh"),
        )
        .with_columns(
            local_relative_power=pl.col("local_power") / pl.col("operating_power_max")
        )
        .select(
            "time_ref",
            "time",
            pl.col("windpark_nve").alias("windpark"),
            "bidding_area",
            "em",
            "local_power",
            "local_relative_power",
            "operating_power_max",
            "mean_production",
            "num_turbines",
            *LOCAL_FEATURES,
        )
        .sort("time_ref", "time", "bidding_area")
    )

    data.sink_parquet("data/windpower_local_dataset.parquet")


if __name__ == "__main__":
    main()
