from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from wind.model.tabular_models import EWMABaseline


class AreaMeanAggregator:
    def __init__(self, local_preds_path: str):
        self.local_preds_path = local_preds_path

    def predict(self, X: pl.LazyFrame):
        local_preds = (
            pl.scan_parquet(self.local_preds_path)
            .group_by("time_ref", "time", "bidding_area", "em")
            .agg(power_pred=pl.col("local_power_pred").sum())
            .group_by("time_ref", "time", "bidding_area")
            .agg(power_pred=pl.col("power_pred").mean())
        )
        return X.join(local_preds, on=["time_ref", "time", "bidding_area"]).select(
            "time_ref", "time", "bidding_area", "power_pred"
        )


def eval_cv(area_data):
    start_dates = [datetime(2025, 1, 1), datetime(2025, 2, 1), datetime(2025, 3, 1)]
    end_dates = [*start_dates[1:], None]
    for val_start, val_end in zip(start_dates, end_dates):
        area_train = area_data.filter(
            pl.col("time_ref") < val_start, pl.col("power").is_not_null()
        )
        if val_end is not None:
            area_val = area_data.filter(
                pl.col("time_ref") >= val_start, pl.col("time_ref") < val_end
            )
        else:
            area_val = area_data.filter(pl.col("time_ref") >= val_start)
        yield area_train, area_val


def per_observation_crps(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_pred.shape[1:] != (1,) * (y_pred.ndim - y_true.ndim - 1) + y_true.shape:
        raise ValueError(
            f"""Expected y_pred to have one extra sample dim on left.
                Actual shapes: {y_pred.shape} versus {y_true.shape}"""
        )

    absolute_error = np.mean(np.abs(y_pred - y_true), axis=0)

    num_samples = y_pred.shape[0]
    if num_samples == 1:
        return absolute_error

    y_pred = np.sort(y_pred, axis=0)
    diff = y_pred[1:] - y_pred[:-1]
    weight = np.arange(1, num_samples) * np.arange(num_samples - 1, 0, -1)
    weight = weight.reshape(weight.shape + (1,) * (diff.ndim - 1))

    return absolute_error - np.sum(diff * weight, axis=0) / num_samples**2


def crps(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
):
    return np.average(per_observation_crps(y_true, y_pred), weights=sample_weight)


def evaluate_single_model_mean():
    area_power_pred = (
        pl.scan_parquet("data/single_model_pred.parquet")
        .group_by("time_ref", "time", "bidding_area", "em")
        .agg(power_pred=pl.col("local_power_pred").sum())
        .group_by("time_ref", "time", "bidding_area")
        .agg(power_pred=pl.col("power_pred").mean())
    )
    area_power_true = (
        pl.scan_parquet("data/wind_power_per_bidzone.parquet").rename(
            {"__index_level_0__": "time"}
        )
    ).unpivot(index="time", variable_name="bidding_area", value_name="power")

    preds = []
    for eval_date in pl.date_range(
        datetime(2025, 1, 1), datetime(2025, 3, 1), interval="1d", eager=True
    ):
        eval_start = eval_date + timedelta(days=1)
        eval_end = eval_start + timedelta(days=1)
        y_pred = area_power_pred.filter(
            pl.col("time_ref").dt.date() == eval_date,
            pl.col("time") >= eval_start,
            pl.col("time") < eval_end,
        ).select("time_ref", "time", "bidding_area", "power_pred")
        y_true = area_power_true.filter(
            pl.col("time") >= eval_start,
            pl.col("time") < eval_end,
        ).select("time", "bidding_area", "power")
        eval_data = y_pred.join(y_true, on=["time", "bidding_area"])
        preds.append(eval_data)

    df_pred = pl.concat(preds).collect()
    print(
        df_pred.group_by("bidding_area")
        .agg(RMSE=((pl.col("power_pred") - pl.col("power")) ** 2).mean().sqrt())
        .sort("bidding_area")
    )
    print(
        df_pred.select(
            ((pl.col("power_pred") - pl.col("power")) ** 2).mean().sqrt()
        ).item()
    )


def evaluate_ewma_baseline():
    area_power_true = (
        (
            pl.scan_parquet("data/wind_power_per_bidzone.parquet").rename(
                {"__index_level_0__": "time"}
            )
        )
        .unpivot(index="time", variable_name="bidding_area", value_name="power")
        .sort("time")
    )

    preds = []
    for eval_date in pl.date_range(
        datetime(2025, 1, 1), datetime(2025, 3, 1), interval="1d", eager=True
    ):
        eval_start = eval_date + timedelta(days=1)
        eval_end = eval_start + timedelta(days=1)
        df_train = area_power_true.filter(pl.col("time") < eval_start).collect()
        df_val = area_power_true.filter(
            pl.col("time") >= eval_start, pl.col("time") < eval_end
        ).collect()

        model = EWMABaseline(span=1)
        model.fit(df_train, "power", ["bidding_area"])
        y_pred = model.predict(df_val)
        y_true = df_val.select("time", "bidding_area", "power")
        eval_data = y_true.with_columns(power_pred=y_pred)
        preds.append(eval_data)

    df_pred = pl.concat(preds)
    print(
        df_pred.group_by("bidding_area")
        .agg(RMSE=((pl.col("power_pred") - pl.col("power")) ** 2).mean().sqrt())
        .sort("bidding_area")
    )
    print(
        df_pred.select(
            ((pl.col("power_pred") - pl.col("power")) ** 2).mean().sqrt()
        ).item()
    )


def evaluate_em0_model_mean():
    area_power_true = (
        pl.scan_parquet("data/wind_power_per_bidzone.parquet").rename(
            {"__index_level_0__": "time"}
        )
    ).unpivot(index="time", variable_name="bidding_area", value_name="power")

    preds = []
    for eval_date in pl.date_range(
        datetime(2025, 1, 1), datetime(2025, 3, 1), interval="1d", eager=True
    ):
        eval_start = eval_date + timedelta(days=1)
        eval_end = eval_start + timedelta(days=1)
        eval_data = area_power_true.filter(
            pl.col("time") >= eval_start,
            pl.col("time") < eval_end,
        )
        preds.append(eval_data)

    df_pred = pl.concat(preds).collect()
    print(
        df_pred.group_by("bidding_area")
        .agg(RMSE=((pl.col("power_pred") - pl.col("power")) ** 2).mean().sqrt())
        .sort("bidding_area")
    )
    print(
        df_pred.select(
            ((pl.col("power_pred") - pl.col("power")) ** 2).mean().sqrt()
        ).item()
    )


def evaluate_bayesian():
    dataset_path = "data/windpower_area_dataset.parquet"
    scaling = (
        pl.scan_parquet(dataset_path)
        .select(
            "bidding_area",
            pl.col("time_ref").cast(pl.Datetime("us")),
            pl.col("time").cast(pl.Datetime("us")),
            "operating_power_max",
        )
        .collect()
    )

    df_pred = (
        pl.read_csv("data/quantile_pred.csv", try_parse_dates=True)
        .join(scaling, on=["bidding_area", "time_ref", "time"])
        .with_columns(power_pred=pl.col("pred_mean") * pl.col("operating_power_max"))
    )
    print(
        df_pred.group_by("bidding_area")
        .agg(RMSE=((pl.col("power_pred") - pl.col("power")) ** 2).mean().sqrt())
        .sort("bidding_area")
    )
    print(
        df_pred.select(
            ((pl.col("power_pred") - pl.col("power")) ** 2).mean().sqrt()
        ).item()
    )


print("EWMA Baseline")
evaluate_ewma_baseline()
print("Single Model Mean")
evaluate_single_model_mean()
# print("EM0 Model Mean")
# evaluate_em0_model_mean()
print("Bayesian Model Mean")
evaluate_bayesian()
