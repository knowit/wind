import numpy as np
import polars as pl

ENSEMBLE_MEMBERS = list(range(15))

AREA_FEATURES = [
    "last_power",
    "recent_mean",
    "ramp",
    "recent_max",
    "recent_min",
    "recent_range",
    "recent_std",
    "ELSPOT NO1",
    "ELSPOT NO2",
    "ELSPOT NO3",
    "ELSPOT NO4",
    "sin_hod",
    "cos_hod",
    "sin_doy",
    "cos_doy",
    "lt",
]

EMOS_FEATURES = [
    "mean_sum_pred",
    "std_sum_pred",
    "min_sum_pred",
    "max_sum_pred",
]


def get_area_capacity(path, times):
    max_capacity = pl.scan_csv(path, try_parse_dates=True)

    area_capacity = (
        times.join(max_capacity, how="cross")
        .filter(pl.col("time") >= pl.col("prod_start_new"))
        .group_by(pl.col("time").alias("time_ref"), "bidding_area")
        .agg(
            operating_power_max=pl.col("operating_power_max").sum(),
            mean_production=pl.col("mean_production").sum(),
            num_turbines=pl.col("num_turbines").sum(),
        )
    )
    return area_capacity


def get_emos_dataset():
    local_pred_path = "data/em0_model_pred.parquet"
    windpower = (
        pl.scan_parquet("data/wind_power_per_bidzone.parquet").rename(
            {"__index_level_0__": "time"}
        )
    ).unpivot(index="time", variable_name="bidding_area", value_name="power")

    times = windpower.select(pl.col("time").unique())
    area_capacity = get_area_capacity("data/windparks_enriched.csv", times)

    windpower = windpower.join(
        area_capacity,
        left_on=["time", "bidding_area"],
        right_on=["time_ref", "bidding_area"],
    ).with_columns(
        relative_power=pl.col("power") / pl.col("operating_power_max"),
    )

    recent_window = 5
    windpower_features = (
        windpower.rename({"time": "time_ref"})
        .sort("time_ref")
        .with_columns(
            last_power="relative_power",
            recent_mean=pl.col("relative_power")
            .rolling_mean(recent_window)
            .over("bidding_area"),
            recent_max=pl.col("relative_power")
            .rolling_max(recent_window)
            .over("bidding_area"),
            recent_min=pl.col("relative_power")
            .rolling_min(recent_window)
            .over("bidding_area"),
            recent_std=pl.col("relative_power")
            .rolling_min(recent_window)
            .over("bidding_area"),
            ramp=pl.col("relative_power")
            - pl.col("relative_power").shift().over("bidding_area"),
        )
        .with_columns(recent_range=pl.col("recent_max") - pl.col("recent_min"))
        .select(
            "time_ref",
            "bidding_area",
            "last_power",
            "recent_mean",
            "ramp",
            "recent_max",
            "recent_min",
            "recent_range",
            "recent_std",
        )
    )

    aggregated_local_preds = (
        pl.scan_parquet(local_pred_path)
        .group_by("time_ref", "time", "bidding_area", "em")
        .agg(sum_local_pred=pl.col("local_power_pred").sum())
    )

    TAU = 2 * np.pi
    hour_frac = pl.col("time").dt.hour().cast(pl.Float64) / 24.0
    doy = pl.col("time").dt.ordinal_day().cast(pl.Float64)
    doy_frac = (doy - 1 + hour_frac) / 365.2425

    emos_dataset = (
        aggregated_local_preds.join(area_capacity, on=["time_ref", "bidding_area"])
        .join(windpower, on=["time", "bidding_area"])
        .with_columns(
            sum_local_pred=pl.col("sum_local_pred") / pl.col("operating_power_max"),
        )
        .group_by("time_ref", "time", "bidding_area")
        .agg(
            power=pl.col("power").first(),
            relative_power=pl.col("relative_power").first(),
            operating_power_max=pl.col("operating_power_max").first(),
            mean_production=pl.col("mean_production").first(),
            num_turbines=pl.col("num_turbines").first(),
            mean_sum_pred=pl.col("sum_local_pred").mean(),
            std_sum_pred=pl.col("sum_local_pred").std(),
            min_sum_pred=pl.col("sum_local_pred").min(),
            max_sum_pred=pl.col("sum_local_pred").max(),
        )
        .join(windpower_features, on=["time_ref", "bidding_area"])
        .with_columns(
            sin_hod=(TAU * hour_frac).sin(),
            cos_hod=(TAU * hour_frac).cos(),
            sin_doy=(TAU * doy_frac).sin(),
            cos_doy=(TAU * doy_frac).cos(),
            lt=(pl.col("time") - pl.col("time_ref")).dt.total_hours(),
        )
        .with_columns(
            (pl.col("bidding_area") == f"ELSPOT NO{k}").alias(f"ELSPOT NO{k}")
            for k in range(1, 5)
        )
        .sort("time_ref", "time", "bidding_area")
        .select(
            "time_ref",
            "time",
            "bidding_area",
            "power",
            "relative_power",
            "operating_power_max",
            "mean_production",
            "num_turbines",
            *AREA_FEATURES,
            *EMOS_FEATURES,
        )
    )

    return emos_dataset


if __name__ == "__main__":
    dataset = get_emos_dataset()
    dataset.sink_parquet("data/windpower_area_dataset.parquet")
