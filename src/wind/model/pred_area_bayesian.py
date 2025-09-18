from datetime import datetime, timedelta
from typing import Sequence

import jax
import numpy as np
import nutpie
import polars as pl
import pymc as pm
import pytensor.tensor as pt

jax.devices()


class NotFittedError(Exception):
    pass


class BayesianAreaModel:
    def __init__(self) -> None:
        self.model = None
        self.idata = None

    def fit(self, X_mu, X_sigma, y, **sample_kwargs):
        self.model = None
        self.idata = None
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
            idata = pm.sample(
                **sample_kwargs,
                progressbar=True,
                # nuts_sampler="numpyro",
                cores=4,
                nuts_sampler="nutpie",
                # nuts_sampler_kwargs=dict(backend="jax"),
            )

        # compiled_model = nutpie.compile_pymc_model(model, backend="jax")
        # idata = nutpie.sample(compiled_model, **sample_kwargs)
        self.model = model
        self.idata = idata
        return self

    def predict(self, X_mu, X_sigma, q: Sequence | None = None):
        if self.model is None:
            raise NotFittedError("Call .fit before .predict")
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
        samples = y_posterior.stack(sample=("chain", "draw")).values
        if q is None:
            return samples
        else:
            qs = np.quantile(samples, q, axis=0).T
            quantiles = pl.DataFrame(
                qs, schema=[f"q{int(1000 * qq):03d}" for qq in q]
            ).with_columns(
                pred_mean=samples.mean(axis=0),
                pred_std=samples.std(axis=0, ddof=1),
            )
            return quantiles


def get_emos_features(data) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_lt = 62
    X_mu = (
        data.select(
            pl.lit(1).alias("intercept"),
            "mean_sum_pred",
            "min_sum_pred",
            "max_sum_pred",
            "pred_lag1",
            "pred_lead1",
            "last_power",
            "recent_mean",
            "ramp",
            "recent_max",
            "recent_min",
            "unavailable_transmission",
        )
        .cast(pl.Float32)
        .collect()
        .to_numpy(order="c")
    )

    X_sigma = (
        data.select(
            pl.lit(1).alias("intercept"),
            pl.col("std_sum_pred").log().alias("log_std_sum_pred"),
            (pl.col("lt") / max_lt).alias("lt"),
            "mean_sum_pred",
            (pl.col("max_sum_pred") - pl.col("min_sum_pred"))
            .log()
            .alias("log_range_pred"),
            pl.col("recent_std").log().alias("log_recent_std"),
            pl.col("ramp").abs().log().alias("log_ramp"),
            "unavailable_transmission",
        )
        .cast(pl.Float32)
        .collect()
        .to_numpy(order="c")
    )

    y = data.select("relative_power").collect().to_numpy()[:, 0]
    sample_weight = data.select("operating_power_max").collect().to_numpy()[:, 0]
    return X_mu, X_sigma, y, sample_weight
