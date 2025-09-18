from datetime import datetime, timedelta

import optuna
import polars as pl
from xgboost import XGBRegressor

from wind.preprocess.prepare_local_data import LOCAL_FEATURES


def station_loo_benchmark(get_model):
    dataset_path = "data/windpower_local_dataset.parquet"
    features = LOCAL_FEATURES
    val_start_date = datetime(2024, 1, 1, 0, 0)
    target = "local_relative_power"
    weight = "operating_power_max"

    data = pl.scan_parquet(dataset_path).filter(
        pl.col(target).is_not_null(), pl.col("lt") > 0
    )

    data_val = data.filter(
        pl.col("time_ref") >= val_start_date,
        pl.col("time").dt.date() == (pl.col("time_ref") + timedelta(days=2)).dt.date(),
    )
    X_val = data_val.select(features).collect()  # .to_numpy()
    windparks = data_val.select(pl.col("windpark").unique()).collect().to_series()

    results = []
    for i, excluded_windpark in enumerate(windparks):
        data_train = data.filter(pl.col("time_ref") < val_start_date).filter(
            pl.col("em") == 0, pl.col("windpark") != excluded_windpark
        )
        X_train = data_train.select(features).collect()  # .to_numpy()
        y_train = data_train.select(target).collect().to_series()  # .to_numpy()
        w_train = data_train.select(weight).collect().to_series()  # .to_numpy()
        model = get_model()
        model.fit(X_train, y_train, sample_weight=w_train)
        pred = model.predict(X_val)

        loo_result = (
            data_val.select(
                "windpark",
                y_true=target,
                weight=weight,
            )
            .with_columns(y_pred=pred)
            .with_columns(
                y_true_scaled=pl.col("y_true") * pl.col("weight"),
                y_pred_scaled=pl.col("y_pred") * pl.col("weight"),
            )
            .group_by("windpark", "weight")
            .agg(
                rmse=((pl.col("y_true") - pl.col("y_pred")) ** 2).mean().sqrt(),
                rmse_scaled=((pl.col("y_true_scaled") - pl.col("y_pred_scaled")) ** 2)
                .mean()
                .sqrt(),
            )
            .with_columns(is_excluded=pl.col("windpark") == excluded_windpark)
        ).collect()
        results.append(loo_result)
        # print(
        #     f"{excluded_windpark=:>20} {rmse_control=:6.4f} {rmse_excluded=:6.4f} {rmse_scaled_control=:7.2f} {rmse_scaled_excluded=:7.2f}"
        # )
        print(f"{i=:3} {excluded_windpark=:>25}")
    return pl.concat(results)


def xgb_station_benchmark(study_name):
    study = optuna.load_study(
        study_name=study_name,
        storage="sqlite:///optuna.db",
    )
    hparams = study.best_params
    hparams["n_estimators"] = study.best_trial.user_attrs["n_estimators"]
    hparams.update(study.best_trial.user_attrs["fixed_params"])

    def get_model():
        return XGBRegressor(**hparams)

    results = station_loo_benchmark(get_model)
    return results
