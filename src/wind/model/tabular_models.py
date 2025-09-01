from abc import ABC

def WindModel(ABC):
    @abstractmethod
    def fit(X, y):
        pass

    def predict(X):
        pass

def LocalAreaWindModel(WindModel):
    