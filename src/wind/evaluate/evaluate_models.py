from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl
from statsmodels.tsa.ar_model import AutoReg

from wind.model.pred_area_bayesian import BayesianAreaModel, get_emos_features


@dataclass
class EvalResult:
    name: str
    rmse: float
    crps: float


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
) -> float:
    return float(
        np.average(per_observation_crps(y_true, y_pred), weights=sample_weight)
    )


def crps_df(df: pl.DataFrame, sample_weight_col: str | None = None) -> float:
    y_true = df.get_column("y_true").to_numpy()  # (N,)
    y_pred = df.select("y_pred").to_numpy().T  # (1, N)
    print(len(y_true), y_pred.min(), y_pred.max(), np.isnan(y_pred).sum())
    if sample_weight_col is None:
        sample_weight = None
    else:
        sample_weight = df.get_column(sample_weight_col).to_numpy()
    return crps(y_true, y_pred, sample_weight)


def ma_baseline(y_true: pl.LazyFrame) -> EvalResult:
    """y_true: pl.DataFrame with 5 columns: bidding_area, time_ref, time, lt, y_true"""
    area_power_pred = (
        (
            pl.scan_parquet("data/wind_power_per_bidzone.parquet").rename(
                {"__index_level_0__": "time_ref"}
            )
        )
        .unpivot(index="time_ref", variable_name="bidding_area", value_name="power")
        .sort("time_ref")
        .with_columns(y_pred=pl.col("power").rolling_mean(24).over("bidding_area"))
    )

    df_val = y_true.join(
        area_power_pred, on=["bidding_area", "time_ref"], how="left"
    ).collect()
    rmse = df_val.select(
        ((pl.col("y_true") - pl.col("y_pred")) ** 2).mean().sqrt()
    ).item()
    crps_score = crps_df(df_val)
    return EvalResult("MA Baseline", rmse, crps_score)


def ar2_model(y_true: pl.LazyFrame) -> EvalResult:
    lookback = 200
    pred_start = 39
    pred_end = 62
    lead_times = np.arange(pred_start, pred_end + 1)

    area_power = pl.scan_parquet("data/wind_power_per_bidzone.parquet").rename(
        {"__index_level_0__": "time_ref"}
    )
    preds = []
    for bidding_area in ["ELSPOT NO1", "ELSPOT NO2", "ELSPOT NO3", "ELSPOT NO4"]:
        for time_ref in (
            y_true.select(pl.col("time_ref").unique())
            .sort("time_ref")
            .collect()
            .to_series()
        ):
            X = (
                area_power.filter(pl.col("time_ref") <= time_ref)
                .sort("time_ref")
                .select(bidding_area)
                .tail(lookback)
                .collect()
                .to_numpy()[:, 0]
            )
            print(bidding_area, time_ref)
            ar_model = AutoReg(X, lags=2).fit()
            pred = ar_model.predict(
                start=lookback + pred_start, end=lookback + pred_end
            )
            preds.append(
                pl.DataFrame(
                    {
                        "time_ref": time_ref,
                        "bidding_area": bidding_area,
                        "y_pred": pred,
                        "lt": lead_times,
                    },
                    schema_overrides={"time_ref": pl.Datetime("ns")},
                )
            )
    pred_df = pl.concat(preds).lazy()
    df_val = (
        y_true.join(pred_df, on=["time_ref", "lt", "bidding_area"], how="left")
        .with_columns(y_pred=pl.col("y_pred").fill_null(0))
        .collect()
    )
    rmse = df_val.select(
        ((pl.col("y_true") - pl.col("y_pred")) ** 2).mean().sqrt()
    ).item()
    crps_score = crps_df(df_val)
    return EvalResult("AR(2)", rmse, crps_score)


def evaluate_single_model_mean(y_true: pl.LazyFrame) -> EvalResult:
    area_power_pred = (
        pl.scan_parquet("data/single_model_pred.parquet")
        .group_by("time_ref", "time", "bidding_area", "em")
        .agg(y_pred=pl.col("local_power_pred").sum())
        .group_by("time_ref", "time", "bidding_area")
        .agg(y_pred=pl.col("y_pred").mean())
    )

    df_val = (
        y_true.join(
            area_power_pred, on=["time_ref", "time", "bidding_area"], how="left"
        )
        .with_columns(y_pred=pl.col("y_pred").fill_null(0))
        .collect()
    )
    rmse = df_val.select(
        ((pl.col("y_true") - pl.col("y_pred")) ** 2).mean().sqrt()
    ).item()
    crps_score = crps_df(df_val)
    return EvalResult("Single Model Mean", rmse, crps_score)


def evaluate_em0_model_mean(y_true: pl.LazyFrame) -> EvalResult:
    area_power_pred = (
        pl.scan_parquet("data/em0_model_pred.parquet")
        .group_by("time_ref", "time", "bidding_area", "em")
        .agg(y_pred=pl.col("local_power_pred").sum())
        .group_by("time_ref", "time", "bidding_area")
        .agg(y_pred=pl.col("y_pred").mean())
    )

    df_val = (
        y_true.join(
            area_power_pred, on=["time_ref", "time", "bidding_area"], how="left"
        )
        .with_columns(y_pred=pl.col("y_pred").fill_null(0))
        .collect()
    )
    rmse = df_val.select(
        ((pl.col("y_true") - pl.col("y_pred")) ** 2).mean().sqrt()
    ).item()
    crps_score = crps_df(df_val)
    return EvalResult("EM0 Model Mean", rmse, crps_score)


def evaluate_bayesian(y_true: pl.LazyFrame) -> EvalResult:
    val_cutoff = y_true.select(pl.col("time_ref").min()).collect().item()
    data = pl.scan_parquet("data/windpower_area_dataset.parquet").filter(
        pl.col("time").dt.date() == (pl.col("time_ref") + timedelta(days=2)).dt.date()
    )

    df_train = (
        data.filter(
            pl.col("time_ref") < val_cutoff,
            pl.col("time_ref") >= val_cutoff - timedelta(days=365),
            pl.col("relative_power").is_not_null(),
        )
        .sort("bidding_area", "time_ref", "time")
        .select(
            pl.col("*").fill_null(strategy="forward").over("bidding_area"),
        )
    )

    df_val = (
        y_true.join(data, on=["bidding_area", "time_ref", "time"], how="left")
        .sort("bidding_area", "time_ref", "time")
        .select(
            pl.col("*").fill_null(strategy="forward").over("bidding_area"),
        )
    )

    preds = []
    for bidding_area in ["ELSPOT NO1", "ELSPOT NO2", "ELSPOT NO3", "ELSPOT NO4"]:
        X_mu_train, X_sigma_train, y_train, scaling_factor_train = get_emos_features(
            df_train.filter(pl.col("bidding_area") == bidding_area)
        )
        X_mu_val, X_sigma_val, y_val, scaling_factor_val = get_emos_features(
            df_val.filter(pl.col("bidding_area") == bidding_area)
        )
        print(
            np.isnan(X_mu_train).sum(),
            np.isnan(X_sigma_train).sum(),
            np.isnan(X_mu_val).sum(),
            np.isnan(X_sigma_val).sum(),
        )
        area_model = BayesianAreaModel()
        area_model.fit(X_mu_train, X_sigma_train, y_train, tune=1000, draws=1000)
        samples = area_model.predict(X_mu_val, X_sigma_val) * np.expand_dims(
            scaling_factor_val, 1
        )
        preds.append(samples)

    pred_samples = np.concat(preds, axis=0)
    print(pred_samples.shape)
    pred_mean = np.mean(pred_samples, axis=1)
    y_true_values = df_val.select("y_true").collect().to_numpy()[:, 0]
    rmse_score = np.sqrt(np.mean((y_true_values - pred_mean) ** 2))
    crps_score = crps(y_true_values, pred_samples.T)
    return EvalResult("Bayesian Calibrator", rmse_score, crps_score)


def get_eval_set(
    eval_start: datetime = datetime(2025, 1, 1), eval_stop: datetime | None = None
):
    area_power = (
        pl.scan_parquet("data/wind_power_per_bidzone.parquet").rename(
            {"__index_level_0__": "time"}
        )
    ).unpivot(index="time", variable_name="bidding_area", value_name="y_true")

    y_true = (
        area_power.select("bidding_area", pl.col("time").alias("time_ref"))
        .filter(
            pl.col("time_ref") >= eval_start,
            pl.col("time_ref").dt.hour() == 9,
        )
        .join(area_power, on="bidding_area")
        .filter(
            pl.col("time").dt.date()
            == (pl.col("time_ref") + timedelta(days=2)).dt.date()
        )
        .select(
            "bidding_area",
            "time_ref",
            "time",
            (pl.col("time") - pl.col("time_ref")).dt.total_hours().alias("lt"),
            "y_true",
        )
    )
    return y_true


def run_eval():
    y_true = get_eval_set()
    results = [
        ma_baseline(y_true),
        ar2_model(y_true),
        evaluate_single_model_mean(y_true),
        evaluate_em0_model_mean(y_true),
        evaluate_bayesian(y_true),
    ]
    print(pl.DataFrame(results, schema=["name", "rmse", "crps"]))


if __name__ == "__main__":
    run_eval()
