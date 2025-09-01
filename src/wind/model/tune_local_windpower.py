"""
Optuna example that demonstrates a pruner for XGBoost.

In this example, we optimize the validation accuracy of cancer detection using XGBoost.
We optimize both the choice of booster model and their hyperparameters. Throughout
training of models, a pruner observes intermediate results and stop unpromising trials.

You can run this example as follows:
    $ python xgboost_integration.py

"""

import json
from datetime import datetime

import optuna
import polars as pl
import sklearn.metrics
import xgboost as xgb
from typing import Iterable


def xgb_trial(trial, X_train, X_val, y_train, y_val):
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
    early_stop = xgb.callback.EarlyStopping(
        rounds=10, metric_name="rmse", data_name="validation_0", save_best=True
    )
    model = xgb.XGBRegressor(
        **param, n_estimators=1000, callbacks=[pruning_callback, early_stop]
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    trial.set_user_attr("fixed_params", fixed_params)
    best_iter = getattr(model, "best_iteration", None)
    trial.set_user_attr("n_estimators", int(best_iter) + 1)

    preds = model.predict(X_val)
    rmse = sklearn.metrics.root_mean_squared_error(y_val, preds)
    return rmse

def get_split_data(dataset_path: str, val_start_date: datetime, features: Iterable[str], target: str) -> tuple[pl.DataFrame, pl.Series, pl.DataFrame, pl.Series]:
    df = (
        pl.scan_parquet(dataset_path)
        .filter(pl.col(target).is_not_null(), pl.col("lt") <= 48)
    )
    df_train = df.filter(pl.col("time_ref") < val_start_date)
    df_val = df.filter(pl.col("time_ref") >= val_start_date)
    X_train = df_train.select(features).collect()
    y_train = df_train.select(target).collect().to_series()
    X_val = df_val.select(features).collect()
    y_val = df_val.select(target).collect().to_series()
    return X_train, y_train, X_val, y_val

def tune_xgb_model(X_train, y_train, X_val, y_val, study_name):
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        direction="minimize",
        study_name=study_name,
        storage="sqlite:///optuna.db",
        load_if_exists=False,
    )

    def objective(trial):
        return xgb_trial(trial, X_train, X_val, y_train, y_val)

    study.optimize(objective, n_trials=100)
    print(study.best_trial)

    best_params = study.best_params
    best_params.update(study.best_trial.user_attrs["fixed_params"])
    best_params["n_estimators"] = study.best_trial.user_attrs["n_estimators"]
    study_name = study.study_name
    with open(f"hparams/{study_name}.json", "w") as f:
        json.dump(best_params, f)


def main():

    from prepare_single_model_data import FEATURES
    study_name = "single_model_xgb"
    dataset_path = "data/windpower_single_model_dataset.parquet"
    features = FEATURES

    # from prepare_ensemble_data import SHARED_FEATURES, get_ensemble_member_features
    # study_name = "model_per_em_xgb"
    # dataset_path = "data/windpower_ensemble_dataset.parquet"
    # em = 0
    # features = [*SHARED_FEATURES, *get_ensemble_member_features(em)]

    val_start_date = datetime(2024, 1, 1, 0, 0)
    target = "local_power"
    X_train, y_train, X_val, y_val = get_split_data(dataset_path, val_start_date, features, target)
    tune_xgb_model(X_train, y_train, X_val, y_val, study_name)


if __name__ == "__main__":
    main()
