import json
from datetime import datetime

import optuna
import polars as pl
import sklearn.metrics
import xgboost as xgb

from prepare_area_ensemble_data import get_all_features_for_ensemble_member


def train_trial(trial, X_train, X_val, y_train, y_val):
    fixed_params = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "device": "cuda",
        "tree_method": "hist",
        "learning_rate": 0.02,
    }
    param = {
        **fixed_params,
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "max_depth": trial.suggest_int("max_depth", 1, 9),
        # "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        ),
    }

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical(
            "sample_type", ["uniform", "weighted"]
        )
        param["normalize_type"] = trial.suggest_categorical(
            "normalize_type", ["tree", "forest"]
        )
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    # Add a callback for pruning.
    # pruning_callback = optuna.integration.XGBoostPruningCallback(
    #     trial, "validation_0-rmse"
    # )
    early_stop = xgb.callback.EarlyStopping(
        rounds=10, metric_name="rmse", data_name="validation_0", save_best=True
    )
    model = xgb.XGBRegressor(
        # **param, n_estimators=1000, callbacks=[pruning_callback, early_stop]
        **param,
        n_estimators=1000,
        callbacks=[early_stop],
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    trial.set_user_attr("fixed_params", fixed_params)
    best_iter = getattr(model, "best_iteration", None)
    trial.set_user_attr("n_estimators", int(best_iter) + 1)

    preds = model.predict(X_val)
    rmse = sklearn.metrics.root_mean_squared_error(y_val, preds)
    return rmse


if __name__ == "__main__":
    dataset_path = "data/windpower_area_ensemble_dataset.parquet"
    em = 0
    features = get_all_features_for_ensemble_member(em)
    target = "power"

    cutoff_date = datetime(2024, 1, 1, 0, 0)
    df = (
        pl.read_parquet(dataset_path)
        .filter(pl.col(target).is_not_null())
        .sort("time_ref", "time", "bidding_area")
    )

    df_train = df.filter(pl.col("time_ref") < cutoff_date)
    df_val = df.filter(pl.col("time_ref") >= cutoff_date)
    X_train = df_train.select(features)
    X_val = df_val.select(features)
    y_train = df_train.get_column(target)
    y_val = df_val.get_column(target)

    study = optuna.create_study(
        # pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        direction="minimize",
        study_name="area_power_xgb_1",
        storage="sqlite:///optuna.db",
        load_if_exists=False,
    )

    def objective(trial):
        return train_trial(trial, X_train, X_val, y_train, y_val)

    study.optimize(objective, n_trials=100)
    print(study.best_trial)

    best_params = study.best_params
    best_params["n_estimators"] = study.best_trial.user_attrs["n_estimators"]
    best_params["objective"] = "reg:squarederror"
    best_params["eval_metric"] = "rmse"
    best_params["device"] = "cuda"
    best_params["tree_method"] = "hist"
    study_name = study.study_name
    with open(f"hparams/{study_name}.json", "w") as f:
        json.dump(best_params, f)
