from datetime import datetime


class WindModel:
    def __init__():
        pass

    def fit(X, y):
        pass

    def predict(X):
        pass


def eval_cv():
    start_dates = [datetime(2025, 1, 1), datetime(2025, 2, 1), datetime(2025, 3, 1)]
    end_dates = [*start_dates[1:], None]
    for sd, ed in zip(start_dates, end_dates):
        yield sd, ed


def evaluate_models():
    pass
