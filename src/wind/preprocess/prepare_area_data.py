from datetime import datetime

import polars as pl
import requests

ENSEMBLE_MEMBERS = list(range(15))

AREA_FEATURES = [
    "last_power",
    "recent_mean",
    "ramp",
    "recent_max",
    "recent_min",
    "recent_range",
    "recent_std",
    "lt",
    "unplannedEvent",
    "unavailable_transmission",
]

EMOS_FEATURES = [
    "mean_sum_pred",
    "std_sum_pred",
    "min_sum_pred",
    "max_sum_pred",
    "pred_lag1",
    "pred_lag2",
    "pred_lead1",
    "pred_lead2",
]


def get_area_capacity(path, times):
    max_capacity = pl.scan_csv(path, try_parse_dates=True)

    area_capacity = (
        times.join(max_capacity, how="cross")
        .filter(pl.col("time") >= pl.col("production_start_date"))
        .group_by(pl.col("time").alias("time_ref"), "bidding_area")
        .agg(
            operating_power_max=pl.col("operating_power_max").sum(),
            mean_production=pl.col("mean_production").sum(),
            num_turbines=pl.col("num_turbines").sum(),
        )
    )
    return area_capacity


def add_unavailable_transmission(df: pl.LazyFrame) -> pl.LazyFrame:
    messages = []
    skip = 0
    while True:
        res = requests.get(
            "https://ummapi.nordpoolgroup.com/messages",
            params={
                "limit": 2000,
                "messageTypes": "TransmissionUnavailability",
                "areas": [
                    "10YNO-1--------2",
                    "10YNO-2--------T",
                    "10YNO-3--------J",
                    "10YNO-4--------9",
                ],
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
        print(
            f"Retrieved: {len(content['items'])} ---- Progress: {skip}/{content['total']}"
        )
        if skip >= content["total"]:
            break

    transmission = (
        pl.json_normalize(messages, infer_schema_length=1000)
        .filter(pl.col("messageType") == 3)
        .explode("transmissionUnits")
        .with_columns(
            # pl.col("eventStart").cast(pl.Datetime).alias("transmissionEventStart"),
            # pl.col("eventStop").cast(pl.Datetime).alias("transmissionEventStop"),
            (pl.col("unavailabilityType") == 1).alias("unplannedEvent"),
            pl.col("publicationDate").cast(pl.Datetime),
            inAreaName=pl.col("transmissionUnits").struct.field("inAreaName"),
            outAreaName=pl.col("transmissionUnits").struct.field("outAreaName"),
            installedCapacity=pl.col("transmissionUnits").struct.field(
                "installedCapacity"
            ),
            timePeriods=pl.col("transmissionUnits").struct.field("timePeriods"),
        )
        .filter(pl.col("outAreaName").is_in(["NO1", "NO2", "NO3", "NO4"]))
        .explode("timePeriods")
        .with_columns(
            unavailableCapacity=pl.col("timePeriods").struct.field(
                "unavailableCapacity"
            ),
            availableCapacity=pl.col("timePeriods").struct.field("availableCapacity"),
            eventStart=pl.col("timePeriods")
            .struct.field("eventStart")
            .cast(pl.Datetime),
            eventStop=pl.col("timePeriods").struct.field("eventStop").cast(pl.Datetime),
        )
        .filter(pl.col("eventStop") > datetime(2020, 1, 1))
        .select(
            "outAreaName",
            "inAreaName",
            "unplannedEvent",
            "installedCapacity",
            "unavailableCapacity",
            "availableCapacity",
            "publicationDate",
            "eventStart",
            "eventStop",
        )
        .lazy()
    )

    unavailable_transmission = (
        df.select("time_ref", "time", "bidding_area")
        .unique()
        .join(
            transmission,
            left_on=pl.col("bidding_area").str.tail(3),
            right_on="outAreaName",
            how="left",
        )
        .filter(
            pl.col("time_ref") > pl.col("publicationDate"),
            pl.col("time") >= pl.col("eventStart"),
            pl.col("time") <= pl.col("eventStop"),
        )
        .group_by("time_ref", "time", "bidding_area")
        .agg(
            pl.col("unplannedEvent").max(),
            pl.col("unavailableCapacity").sum(),
            pl.col("installedCapacity").sum(),
        )
        .with_columns(
            unavailable_transmission=pl.col("unavailableCapacity")
            / pl.col("installedCapacity")
        )
    )
    return df.join(
        unavailable_transmission, on=["time_ref", "time", "bidding_area"], how="left"
    ).with_columns(
        pl.col("unplannedEvent").fill_null(False),
        pl.col("unavailable_transmission").fill_null(0),
    )


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

    recent_window = 24
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
        .group_by("time_ref", "time", "lt", "bidding_area", "em")
        .agg(sum_local_pred=pl.col("local_power_pred").sum())
    )

    emos_dataset = (
        aggregated_local_preds.join(area_capacity, on=["time_ref", "bidding_area"])
        .join(windpower, on=["time", "bidding_area"])
        .with_columns(
            sum_local_pred=pl.col("sum_local_pred") / pl.col("operating_power_max"),
        )
        .group_by("time_ref", "time", "lt", "bidding_area")
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
        .with_columns(
            pred_lag1=pl.col("mean_sum_pred")
            .shift(1)
            .over(["time_ref", "bidding_area"], order_by="time"),
            pred_lag2=pl.col("mean_sum_pred")
            .shift(2)
            .over(["time_ref", "bidding_area"], order_by="time"),
            pred_lead1=pl.col("mean_sum_pred")
            .shift(-1)
            .over(["time_ref", "bidding_area"], order_by="time"),
            pred_lead2=pl.col("mean_sum_pred")
            .shift(-2)
            .over(["time_ref", "bidding_area"], order_by="time"),
        )
        .join(windpower_features, on=["time_ref", "bidding_area"])
        .pipe(add_unavailable_transmission)
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


def main():
    dataset = get_emos_dataset()
    dataset.sink_parquet("data/windpower_area_dataset.parquet")


if __name__ == "__main__":
    main()
