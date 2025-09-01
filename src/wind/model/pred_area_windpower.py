import json
from datetime import datetime

import polars as pl
from xgboost import XGBRegressor

from prepare_area_ensemble_data import get_all_features_for_ensemble_member
from prepare_ensemble_data import ENSEMBLE_MEMBERS


def predict_ensemble(df_train, df_val, target, model_params):
    y_train = df_train.get_column(target)
    ensemble_member_preds = {}
    for em in ENSEMBLE_MEMBERS:
        features = get_all_features_for_ensemble_member(em)
        X_train = df_train.select(features)
        X_val = df_val.select(features)
        model = XGBRegressor(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        ensemble_member_preds[f"area_power_pred_{em:02d}"] = y_pred
        print(f" {em:02d}", end="")
    print()

    df_pred = df_val.select(
        "time_ref", "time", "lt", "bidding_area", "power"
    ).with_columns(**ensemble_member_preds)
    return df_pred


def get_hparams(study_name):
    with open(f"hparams/{study_name}.json") as f:
        hparams = json.load(f)
    return hparams


if __name__ == "__main__":
    dataset_path = "data/windpower_area_ensemble_dataset.parquet"
    target = "power"
    cutoff_date = datetime(2024, 1, 1, 0, 0)
    df = (
        pl.read_parquet(dataset_path)
        .filter(pl.col(target).is_not_null())
        .sort("time_ref", "time", "bidding_area")
    )

    df_train = df.filter(pl.col("time_ref") < cutoff_date)
    df_val = df.filter(pl.col("time_ref") >= cutoff_date)

    study_name = "area_power_xgb_1"
    hparams = get_hparams(study_name)

    # preds = predict(data, FEATURES, target, hparams)
    preds = predict_ensemble(df_train, df_val, target, hparams)
    preds.write_parquet("data/area_power_pred.parquet")
