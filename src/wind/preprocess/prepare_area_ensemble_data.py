import polars as pl

from prepare_ensemble_data import ENSEMBLE_MEMBERS

NO_AGG_FEATURES = [
    "ELSPOT NO1",
    "ELSPOT NO2",
    "ELSPOT NO3",
    "ELSPOT NO4",
    "sin_hod",
    "cos_hod",
    "sin_doy",
    "cos_doy",
]

AREA_FEATURES = [
    "operating_power_max",
    "mean_production",
    "num_turbines",
    "location_mean_ws",
]

ENSEMBLE_FEATURES = [
    "ws_power_scaled",
    "ws_turbine_scaled",
    "wind_power_density_scaled",
    "wind_alignment",
    "power_pred",
]


def weighted_mean(variable: str) -> pl.Expr:
    return (
        (pl.col(variable) * pl.col("operating_power_max")).sum()
        / pl.col("operating_power_max").sum()
    ).alias(variable)


def em_mean(variable: str, em: int) -> pl.Expr:
    return pl.col(f"{variable}_{em:02d}").mean().alias(f"{variable}_{em:02d}")


def em_weighted_mean(variable: str, em: int) -> pl.Expr:
    return weighted_mean(f"{variable}_{em:02d}")


def get_ensemble_member_features(em: int) -> list[str]:
    return [f"{feature_name}_{em:02d}" for feature_name in ENSEMBLE_FEATURES]


def get_all_features_for_ensemble_member(em: int) -> list[str]:
    return [*NO_AGG_FEATURES, *AREA_FEATURES, *get_ensemble_member_features(em)]


def power_pred_ewma(em: int, span: int) -> pl.Expr:
    return (
        pl.col(f"power_pred_{em:02d}")
        .ewm_mean(span=span)
        .over("bidding_area", "time", order_by="time_ref")
        .alias(f"power_pred_{em:02d}_ewma{span}")
    )


def power_pred_lag(em: int, lag: int) -> pl.Expr:
    return (
        pl.col(f"power_pred_{em:02d}")
        .shift(lag)
        .over("bidding_area", "time_ref", order_by="lt")
        .alias(f"power_pred_{em:02d}_lag{lag}")
    )


if __name__ == "__main__":
    ensemble_dataset = pl.scan_parquet("data/windpower_ensemble_dataset.parquet")
    ensemble_preds = pl.scan_parquet("data/local_power_pred.parquet")
    area_dataset = (
        ensemble_dataset.join(
            ensemble_preds, on=["time_ref", "time", "sid", "windpark_name"]
        )
        .group_by(
            "time_ref",
            "time",
            "lt",
            "bidding_area",
            "power",
            *NO_AGG_FEATURES,
        )
        .agg(
            pl.col("operating_power_max").sum(),
            pl.col("mean_production").sum(),
            pl.col("num_turbines").sum(),
            weighted_mean("location_mean_ws"),
            *(em_mean("ws_power_scaled", em) for em in ENSEMBLE_MEMBERS),
            *(em_mean("ws_turbine_scaled", em) for em in ENSEMBLE_MEMBERS),
            *(em_mean("wind_power_density_scaled", em) for em in ENSEMBLE_MEMBERS),
            *(em_weighted_mean("wind_alignment", em) for em in ENSEMBLE_MEMBERS),
            *(
                pl.col(f"local_power_pred_{em:02d}").sum().alias(f"power_pred_{em:02d}")
                for em in ENSEMBLE_MEMBERS
            ),
        )
        # .with_columns(
        #     *(power_pred_lag(em, 1) for em in ENSEMBLE_MEMBERS),
        #     *(power_pred_ewma(em, 2) for em in ENSEMBLE_MEMBERS),
        #     *(power_pred_ewma(em, 3) for em in ENSEMBLE_MEMBERS),
        #     *(power_pred_ewma(em, 6) for em in ENSEMBLE_MEMBERS),
        #     *(power_pred_ewma(em, 12) for em in ENSEMBLE_MEMBERS),
        #     *(power_pred_ewma(em, 24) for em in ENSEMBLE_MEMBERS),
        # )
        .sort(
            "time_ref",
            "time",
            "lt",
            "bidding_area",
        )
    )

    area_dataset.sink_parquet("data/windpower_area_ensemble_dataset.parquet")
