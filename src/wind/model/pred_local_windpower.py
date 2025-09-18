import json
from datetime import datetime
from typing import Any, Callable, Iterable, Self

import polars as pl
from sklearn.exceptions import NotFittedError
from xgboost import XGBRegressor

from wind.preprocess.prepare_local_data import ENSEMBLE_MEMBERS, LOCAL_FEATURES


class TimeSeriesLeakage(Exception):
    pass


class LocalPowerModel:
    def __init__(
        self,
        features: Iterable[str],
        target: str,
        weight: str,
        get_estimator: Callable[[], XGBRegressor],
    ):
        self.get_estimator = get_estimator
        self.features = features
        self.target = target
        self.weight = weight
        self.valid_time: datetime | None = None
        self.estimator: XGBRegressor | None = None

    def fit(self, data: pl.LazyFrame, **fit_kwargs) -> Self:
        self.valid_time = data.select(pl.col("time_ref").max()).collect().item()
        X = data.select(self.features).collect()
        y = data.select(self.target).collect().to_series()
        w = data.select(self.weight).collect().to_series()
        self.estimator = self.get_estimator()
        self.estimator.fit(X, y, sample_weight=w, **fit_kwargs)
        return self

    def predict(self, data: pl.LazyFrame) -> pl.Series:
        if self.valid_time >= data.select(pl.col("time_ref").min()).collect().item():
            raise TimeSeriesLeakage(
                "The data passed to predict contains a time_ref before last time_ref in the training data."
            )
        if self.estimator is None:
            raise NotFittedError()

        X_pred = data.select(self.features).collect()
        w_pred = data.select(self.weight).collect().to_series()
        y_pred = self.estimator.predict(X_pred)
        return y_pred * w_pred


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
    train_and_predict: Callable[[pl.LazyFrame, pl.LazyFrame], pl.Series],
) -> pl.DataFrame:
    monthly_preds: list[pl.DataFrame] = []
    for df_train, df_pred in iter_time_blocks(
        data, target, first_pred_time=datetime(2023, 1, 1)
    ):
        y_pred = train_and_predict(df_train, df_pred)
        monthly_preds.append(
            df_pred.select(*id_cols).collect().with_columns(local_power_pred=y_pred)
        )
    return pl.concat(monthly_preds)


def train_predict_single_model(
    features: Iterable[str], target: str, hparams: dict[str, Any]
) -> Callable[[pl.LazyFrame, pl.LazyFrame], dict[str, pl.Series]]:
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


def train_predict_model_per_em(
    features: Iterable[str], target: str, hparams: dict[str, Any]
) -> Callable[[pl.LazyFrame, pl.LazyFrame], pl.LazyFrame]:
    def train_and_predict(
        df_train: pl.LazyFrame, df_pred: pl.LazyFrame
    ) -> pl.LazyFrame:
        preds: list[pl.LazyFrame] = []
        for em in ENSEMBLE_MEMBERS:
            y_train = (
                df_train.filter(pl.col("em") == em).select(target).collect().to_series()
            )
            X_train = df_train.filter(pl.col("em") == em).select(features).collect()
            X_pred = df_pred.filter(pl.col("em") == em).select(features).collect()
            model = XGBRegressor(**hparams)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_pred)
            em_pred = pl.DataFrame(dict(em=em, local_power_pred=y_pred)).lazy()
            preds.append(em_pred)
            print(f"{em:02d}", end=" ", flush=True)
        return pl.concat(preds)

    return train_and_predict


def train_predict_em0_model(
    features: Iterable[str], target: str, weight: str, hparams: dict[str, Any]
) -> Callable[[pl.LazyFrame, pl.LazyFrame], pl.Series]:
    def train_and_predict(df_train: pl.LazyFrame, df_pred: pl.LazyFrame) -> pl.Series:
        df_train = df_train.filter(pl.col("em") == 0)
        X_train = df_train.select(features).collect()
        X_pred = df_pred.select(features).collect()
        y_train = df_train.select(target).collect().to_series()
        w_train = df_train.select(weight).collect().to_series()
        w_pred = df_pred.select(weight).collect().to_series()
        model = XGBRegressor(**hparams)
        model.fit(X_train, y_train, sample_weight=w_train)
        y_pred = model.predict(X_pred)
        return y_pred * w_pred

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


def main():
    target = "local_relative_power"
    weight = "operating_power_max"
    dataset_path = "data/windpower_local_dataset.parquet"
    study_name = "em0_geo_features_xgb_3"
    output_path = "data/em0_model_pred.parquet"
    id_cols = [
        "time_ref",
        "time",
        "lt",
        "windpark",
        "bidding_area",
        "em",
    ]
    hparams = get_hparams(study_name)
    data = pl.scan_parquet(dataset_path)
    train_and_predict = train_predict_em0_model(LOCAL_FEATURES, target, weight, hparams)

    preds = rolling_monthly_prediction(data, target, id_cols, train_and_predict)
    preds.write_parquet(output_path)


if __name__ == "__main__":
    main()
