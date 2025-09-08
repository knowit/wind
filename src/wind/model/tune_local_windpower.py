import json
from datetime import datetime
from typing import Iterable

import optuna
import polars as pl
import sklearn.metrics
import xgboost as xgb

from wind.preprocess.prepare_local_data import LOCAL_FEATURES


def xgb_trial(trial, X_train, X_val, y_train, y_val, w_train, w_val):
    fixed_params = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "device": "cuda",
        "tree_method": "hist",
        "learning_rate": 0.1,
    }
    param = {
        **fixed_params,
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "max_depth": trial.suggest_int("max_depth", 1, 9),
        # "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        ),
    }

    # Add a callback for pruning.
    pruning_callback = optuna.integration.XGBoostPruningCallback(
        trial, "validation_0-rmse"
    )
    early_stop = xgb.callback.EarlyStopping(  # type: ignore
        rounds=10, metric_name="rmse", data_name="validation_0", save_best=True
    )
    model = xgb.XGBRegressor(
        **param, n_estimators=1000, callbacks=[pruning_callback, early_stop]
    )
    model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)])

    trial.set_user_attr("fixed_params", fixed_params)
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is None:
        raise ValueError(
            "No iterations found to select best iteration. Something likely went wrong during training."
        )
    trial.set_user_attr("n_estimators", int(best_iter) + 1)

    preds = model.predict(X_val)
    rmse = sklearn.metrics.root_mean_squared_error(y_val, preds, sample_weight=w_val)
    return rmse


def get_split_data(
    data: pl.LazyFrame,
    val_start_date: datetime,
    features: Iterable[str],
    target: str,
    weight: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series, pl.Series, pl.Series]:
    data_train = data.filter(pl.col("time_ref") < val_start_date).filter(
        pl.col("em") == 0
    )
    data_val = data.filter(pl.col("time_ref") >= val_start_date)

    X_train = data_train.select(features).collect()
    X_val = data_val.select(features).collect()

    y_train = data_train.select(target).collect().to_series()
    y_val = data_val.select(target).collect().to_series()

    if weight is None:
        w_train = pl.ones(y_train.len(), eager=True)
        w_val = pl.ones(y_val.len(), eager=True)
    else:
        w_train = data_train.select(weight).collect().to_series()
        w_val = data_val.select(weight).collect().to_series()
    return X_train, X_val, y_train, y_val, w_train, w_val


def tune_xgb_model(X_train, X_val, y_train, y_val, w_train, w_val, study_name):
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=25),
        direction="minimize",
        study_name=study_name,
        storage="sqlite:///optuna.db",
        load_if_exists=False,
    )

    def objective(trial):
        return xgb_trial(trial, X_train, X_val, y_train, y_val, w_train, w_val)

    study.optimize(objective, n_trials=100)
    print(study.best_trial)

    best_params = study.best_params
    best_params.update(study.best_trial.user_attrs["fixed_params"])
    best_params["n_estimators"] = study.best_trial.user_attrs["n_estimators"]
    study_name = study.study_name
    with open(f"hparams/{study_name}.json", "w") as f:
        json.dump(best_params, f)


def main():
    dataset_path = "data/windpower_local_dataset.parquet"
    features = LOCAL_FEATURES
    val_start_date = datetime(2024, 1, 1, 0, 0)
    target = "local_relative_power"
    weight = "operating_power_max"

    study_name = "em0_model_xgb_2"
    data = pl.scan_parquet(dataset_path).filter(
        pl.col(target).is_not_null(), pl.col("lt") <= 48
    )

    X_train, X_val, y_train, y_val, w_train, w_val = get_split_data(
        data, val_start_date, features, target, weight
    )
    tune_xgb_model(X_train, X_val, y_train, y_val, w_train, w_val, study_name)


def main_area():
    from wind.preprocess.prepare_area_data import AREA_FEATURES

    dataset_path = "data/windpower_area_dataset.parquet"
    features = AREA_FEATURES
    val_start_date = datetime(2025, 1, 1, 0, 0)
    target = "relative_power"
    weight = "operating_power_max"

    study_name = "em0_area_model_xgb_2"
    data = pl.scan_parquet(dataset_path).filter(
        pl.col(target).is_not_null(), pl.col("lt") <= 48
    )

    X_train, X_val, y_train, y_val, w_train, w_val = get_split_data(
        data, val_start_date, features, target, weight
    )
    tune_xgb_model(X_train, X_val, y_train, y_val, w_train, w_val, study_name)


if __name__ == "__main__":
    main()
    # main_area()
