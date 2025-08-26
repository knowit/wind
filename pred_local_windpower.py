from datetime import datetime

import optuna
import polars as pl
from xgboost import XGBRegressor

from prepare_data import FEATURES


def predict(data, features, target, model_params):
    monthly_time_ref = (
        data.filter(
            pl.col("time_ref") > datetime(2021, 1, 1), pl.col(target).is_not_null()
        )
        .group_by(month_ref=pl.col("time_ref").dt.strftime("%Y-%m"))
        .agg(time_ref=pl.col("time_ref").first())
        # .collect()
        .get_column("time_ref")
        .sort()
    )
    preds = []
    for pred_start, pred_end in zip(monthly_time_ref, monthly_time_ref.shift(-1)):
        print(pred_start.strftime("%Y-%m-%d"))
        df_train = data.filter(
            pl.col("time_ref") < pred_start, pl.col(target).is_not_null()
        )
        if pred_end is not None:
            df_pred = data.filter(
                pl.col("time_ref") >= pred_start, pl.col("time_ref") < pred_end
            )
        else:
            df_pred = data.filter(pl.col("time_ref") >= pred_start)
        # df_train = df_train.collect(gpu=True)
        # df_pred = df_pred.collect(gpu=True)
        X_train = df_train.select(features)
        y_train = df_train.get_column(target)
        X_pred = df_pred.select(features)
        model = XGBRegressor(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_pred)
        preds.append(
            df_pred.with_columns(local_power_pred=y_pred).select(
                "time_ref", "time", "windpark_name", "local_power_pred"
            )
        )
    return pl.concat(preds)


if __name__ == "__main__":
    data = pl.read_parquet("data/windpower_dataset.parquet")
    target = "local_power"

    study = optuna.load_study(
        study_name="local_power_xgb_6",
        storage="sqlite:///optuna.db",
    )
    best_params = study.best_params
    best_params["n_estimators"] = study.best_trial.user_attrs["n_estimators"]
    best_params["objective"] = "reg:squarederror"
    best_params["eval_metric"] = "rmse"
    best_params["device"] = "cuda"
    best_params["tree_method"] = "hist"

    preds = predict(data, FEATURES, target, best_params)
    preds.write_parquet("data/local_power_pred.parquet")
