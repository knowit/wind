from typing import Iterable, Self

import polars as pl


class EWMABaseline:
    def __init__(self, span: int):
        self.span = span
        self.target = None
        self.groups = None
        self.ewma = pl.DataFrame

    def fit(self, X, target: str, groups: Iterable[str]) -> Self:
        self.target = target
        if isinstance(groups, str):
            self.groups = [groups]
        else:
            self.groups = groups
        self.ewma = X.group_by(self.groups).agg(
            y_pred=pl.col(target).tail(self.span * 2).ewm_mean(span=self.span).last()
        )
        return self

    def predict(self, X) -> pl.Series:
        y_pred = X.join(self.ewma, on=self.groups).get_column("y_pred")
        return y_pred
