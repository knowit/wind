import json
from datetime import datetime
from typing import Any, Callable, Iterable

import polars as pl
from xgboost import XGBRegressor

from wind.preprocess.prepare_ensemble_data import ENSEMBLE_MEMBERS, get_all_features_for_ensemble_member

def iter_time_blocks(
    data: pl.LazyFrame, target_col: str, first_pred_time: datetime | None = None
) -> Iterable[tuple[pl.LazyFrame, pl.LazyFrame]]:
    time_ref = data.filter(pl.col(target_col).is_not_null()).select("time_ref")

    if first_pred_time is not None:
        time_ref = time_ref.filter(pl.col("time_ref") >= first_pred_time)

    monthly_time_ref = (
        time_ref.group_by(month_ref=pl.col("time_ref").dt.strftime("%Y-%m"))
        .agg(time_ref=pl.col("time_ref").first())
        .select("time_ref")
        .sort("time_ref")
        .collect()
        .to_series()
    )

    for pred_start, pred_end in zip(monthly_time_ref, monthly_time_ref.shift(-1)):
        print(pred_start.strftime("%Y-%m-%d"))
        df_train = data.filter(
            pl.col("time_ref") < pred_start, pl.col(target_col).is_not_null()
        )
        if pred_end is not None:
            df_pred = data.filter(
                pl.col("time_ref") >= pred_start, pl.col("time_ref") < pred_end
            )
        else:
            df_pred = data.filter(pl.col("time_ref") >= pred_start)

        yield df_train, df_pred


def rolling_monthly_prediction(
    data: pl.LazyFrame,
    target: str,
    id_cols: Iterable[str],
    train_and_predict: Callable[[pl.LazyFrame, pl.LazyFrame], dict[str, pl.Series]],
) -> pl.DataFrame:
    monthly_preds: list[pl.DataFrame] = []
    for df_train, df_pred in iter_time_blocks(
        data, target, first_pred_time=datetime(2023, 1, 1)
    ):
        y_pred = train_and_predict(df_train, df_pred)
        monthly_preds.append(df_pred.select(*id_cols).collect().with_columns(**y_pred))
    return pl.concat(monthly_preds)


def train_predict_single_model(features: Iterable[str], target: str, hparams: dict[str, Any]) -> Callable[[pl.LazyFrame, pl.LazyFrame], dict[str, pl.Series]]:
    def train_and_predict(
        df_train: pl.LazyFrame, df_pred: pl.LazyFrame
    ) -> dict[str, pl.Series]:
        y_train = df_train.select(target).collect().to_series()
        X_train = df_train.select(features).collect()
        X_pred = df_pred.select(features).collect()
        model = XGBRegressor(**hparams)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_pred)
        return {"local_power_pred": y_pred}

    return train_and_predict


def train_predict_model_per_em(target: str, hparams: dict[str, Any]) -> Callable[[pl.LazyFrame, pl.LazyFrame], dict[str, pl.Series]]:
    def train_and_predict(
        df_train: pl.LazyFrame, df_pred: pl.LazyFrame
    ) -> dict[str, pl.Series]:
        y_train = df_train.select(target).collect().to_series()
        ensemble_member_preds: dict[str, pl.Series] = {}
        for em in ENSEMBLE_MEMBERS:
            features = get_all_features_for_ensemble_member(em)
            X_train = df_train.select(features).collect()
            X_pred = df_pred.select(features).collect()
            model = XGBRegressor(**hparams)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_pred)
            ensemble_member_preds[f"local_power_pred_{em:02d}"] = y_pred
            print(f"{em:02d}", end=" ", flush=True)
        return ensemble_member_preds

    return train_and_predict


def get_hparams(study_name: str) -> dict[str, Any]:
    try:
        with open(f"hparams/{study_name}.json") as f:
            hparams = json.load(f)
    except FileNotFoundError:
        import optuna

        study = optuna.load_study(
            study_name=study_name,
            storage="sqlite:///optuna.db",
        )
        hparams = study.best_params
        hparams["n_estimators"] = study.best_trial.user_attrs["n_estimators"]
        hparams.update(study.best_trial.user_attrs["fixed_params"])
    return hparams


if __name__ == "__main__":
    prediction_horizon = 48
    target = "local_power"

    # ---- Single Model ----
    from wind.preprocess.prepare_single_model_data import FEATURES

    dataset_path = "data/windpower_single_model_dataset.parquet"
    study_name = "single_model_xgb_1"
    output_path = "data/single_model_pred.parquet"
    id_cols = ["time_ref", "time", "sid", "windpark_name", "bidding_area", "em"]
    hparams = get_hparams(study_name)
    data = pl.scan_parquet(dataset_path).filter(pl.col("lt") <= prediction_horizon)
    train_and_predict = train_predict_single_model(FEATURES, target, hparams)

    # # ---- Model Per EM ----
    # dataset_path = "data/windpower_ensemble_dataset.parquet"
    # study_name = "local_power_48h_xgb_0"
    # output_path = "data/model_per_em_pred.parquet"
    # id_cols = ["time_ref", "time", "sid", "windpark_name", "bidding_area"]
    # hparams = get_hparams(study_name)
    # data = pl.scan_parquet(dataset_path).filter(pl.col("lt") <= prediction_horizon)
    # train_and_predict = train_predict_model_per_em(target, hparams)

    preds = rolling_monthly_prediction(data, target, id_cols, train_and_predict)
    preds.write_parquet(output_path)
