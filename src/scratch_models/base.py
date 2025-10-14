import numpy as np
from abc import ABC, abstractmethod

class BaseClassifier(ABC):
    def __init__(self):
        self.is_fitted = False
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_ = None

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def score(self, X, y):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def _validate_input(self, X, y=None):
        X = np.asarray(X)

        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf values")
        
        if y is not None:
            y = np.asarray(y)
            if not np.isfinite(y).all():
                raise ValueError("y contains NaN or Inf values")
            
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"X and y have incompatible shapes: "
                               f"{X.shape[0]} vs {y.shape[0]}")
            
            return X, y
        
        return X

    def _initialise_classes(self, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

    def __repr__(self):
        params = ', '.join([f"{k}={v}" for k, v in self.__dict__.items() 
                           if not k.endswith('_')])
        return f"{self.__class__.__name__}({params})"