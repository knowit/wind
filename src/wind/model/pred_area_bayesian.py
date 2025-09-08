from datetime import datetime, timedelta

import jax
import numpy as np
import nutpie
import polars as pl
import pymc as pm
import pytensor.tensor as pt

jax.devices()


class BayesianAreaModel:
    def __init__(self) -> None:
        pass

    def fit(self, X_mu, X_sigma, y, **sample_kwargs):
        with pm.Model() as model:
            Xmu = pm.Data("Xmu", X_mu)
            Xsig = pm.Data("Xsig", X_sigma)
            y_obs = pm.Data("y_obs", y)

            # Priors
            beta = pm.Normal("beta", mu=0.0, sigma=2.0, shape=X_mu.shape[1])
            gamma = pm.Normal("gamma", mu=0.0, sigma=0.2, shape=X_sigma.shape[1])

            # Linear predictors
            mu = pm.Deterministic("mu", pt.dot(Xmu, beta))  # logit-mean
            log_sig = pm.Deterministic("log_sigma", pt.dot(Xsig, gamma))
            sigma = pm.Deterministic("sigma", pt.exp(log_sig))  # > 0

            pm.LogitNormal("y", mu=mu, sigma=sigma, observed=y_obs)

        compiled_model = nutpie.compile_pymc_model(model, backend="jax")
        idata = nutpie.sample(compiled_model, **sample_kwargs)
        # idata = pm.sample(
        #     **sample_kwargs,
        #     progressbar=True,
        #     nuts_sampler="numpyro",
        #     # nuts_sampler="nutpie",
        #     # nuts_sampler_kwargs=dict(backend="jax"),
        # )
        self.model = model
        self.idata = idata
        return self

    def predict(self, X_mu, X_sigma, q=(0.025, 0.05, 0.5, 0.95, 0.975)):
        with self.model:
            pm.set_data(
                {
                    "Xmu": X_mu,
                    "Xsig": X_sigma,
                    "y_obs": np.zeros(X_mu.shape[0], dtype=np.float32),
                }
            )
            ppc = pm.sample_posterior_predictive(self.idata)

        y_posterior = ppc.posterior_predictive["y"]  # type: ignore
        samples = (
            y_posterior.stack(sample=("chain", "draw"))
            .transpose("sample", "y_dim_0")
            .values
        )
        # Quantiles per observation
        qs = np.quantile(samples, q, axis=0).T
        quantiles = pl.DataFrame(
            qs, schema=[f"q{int(1000 * qq):03d}" for qq in q]
        ).with_columns(
            pred_mean=samples.mean(axis=0),
            pred_std=samples.std(axis=0, ddof=1),
        )
        return quantiles, samples


def get_emos_features(data):
    max_lt = 48
    X_mu = (
        data.select(
            pl.lit(1).alias("intercept"),
            "mean_sum_pred",
            "min_sum_pred",
            "max_sum_pred",
            "last_power",
            "recent_mean",
            "ramp",
            "recent_max",
            "recent_min",
            "sin_hod",
            "cos_hod",
            "sin_doy",
            "cos_doy",
        )
        .cast(pl.Float32)
        .collect()
        .to_numpy()
    )

    X_sigma = (
        data.select(
            pl.lit(1).alias("intercept"),
            pl.col("std_sum_pred").log().alias("log_std_sum_pred"),
            (pl.col("max_sum_pred") - pl.col("min_sum_pred"))
            .log()
            .alias("log_max_sum_pred"),
            pl.col("recent_std").log().alias("log_recent_std"),
            pl.col("ramp").abs().log().alias("log_ramp"),
            "sin_hod",
            "cos_hod",
            "sin_doy",
            "cos_doy",
            (pl.col("lt") / max_lt).alias("lt"),
        )
        .cast(pl.Float32)
        .collect()
        .to_numpy()
    )

    y = data.select("relative_power").collect().to_numpy()[:, 0]
    scaling_factor = data.select("operating_power_max").collect().to_numpy()[:, 0]
    return X_mu, X_sigma, y, scaling_factor


dataset_path = "data/windpower_area_dataset.parquet"
val_cutoff = datetime(2025, 1, 1)
data = pl.scan_parquet(dataset_path).filter(
    pl.col("time") >= pl.col("time_ref").dt.date() + timedelta(days=1),
    pl.col("time") < pl.col("time_ref").dt.date() + timedelta(days=2),
)


preds = []
for bidding_area in ["ELSPOT NO1", "ELSPOT NO2", "ELSPOT NO3", "ELSPOT NO4"]:
    data_train = data.filter(
        pl.col("bidding_area") == bidding_area,
        pl.col("time_ref") < val_cutoff,
        pl.col("time_ref") >= val_cutoff - timedelta(days=365),
        pl.col("relative_power").is_not_null(),
    )
    data_val = data.filter(
        pl.col("bidding_area") == bidding_area,
        pl.col("time_ref") >= val_cutoff,
    )
    X_mu_train, X_sigma_train, y_train, scaling_factor_train = get_emos_features(
        data_train
    )
    print(
        bidding_area,
        f"train samples: {data_train.collect().height}",
        f"val samples: {data_val.collect().height}",
    )
    X_mu_val, X_sigma_val, y_val, scaling_factor_val = get_emos_features(data_val)
    area_model = BayesianAreaModel()
    area_model.fit(X_mu_train, X_sigma_train, y_train)
    quantiles, samples = area_model.predict(X_mu_val, X_sigma_val)
    area_pred = pl.concat(
        [
            data_val.select(
                "bidding_area",
                "time_ref",
                "time",
                "lt",
                "relative_power",
                "operating_power_max",
            ).collect(),
            quantiles,
        ],
        how="horizontal",
    )
    preds.append(area_pred)

pl.concat(preds).write_csv("data/quantile_pred.csv")
