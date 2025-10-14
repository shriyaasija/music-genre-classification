import numpy as np
from .base import BaseClassifier

class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=0.01, verbose=False, tol=1e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.verbose = verbose
        self.tol = tol
        
        # Will be set during fit
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        X, y = self._validate_input(X, y)

        n_samples, n_features = X.shape
        self._initialise_classes(y)
        self.n_features_ = n_features

        self.weights = np.random.randn(self.n_classes_, n_features) * 0.01
        self.bias = np.zeros(self.n_classes_)

        y_encoded = self._one_hot_encode(y)

        for iteration in range(self.n_iterations):
            z = X @ self.weights.T + self.bias

            y_pred = self._softmax(z)

            loss = self._calculate_loss(y_encoded, y_pred)
            self.loss_history.append(loss)

            error = y_pred - y_encoded

            dw = (1/n_samples) * (error.T @ X) + (self.regularization * self.weights)
            db = (1/n_samples) * np.sum(error, axis=0)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if self.verbose and (iteration % 100 == 0):
                accuracy = np.mean(np.argmax(y_pred, axis=1) == y)
                print(f"Iteration {iteration}: Loss = {loss:.4f}, "
                      f"Accuracy = {accuracy:.4f}")
                
                if iteration > 0 and abs(self.loss_history[-1] - 
                                    self.loss_history[-2]) < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {iteration}")
                    break
            
            self.is_fitted = True
            return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._validate_input(X)

        probabilities = self.predict_probabilities(X)

        predictions = np.argmax(probabilities, axis=1)

        return self.classes_[predictions]
    
    def predict_probabilities(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        
        X = self._validate_input(X)

        z = X @ self.weights.T + self.bias

        probabilities = self._softmax(z)

        return probabilities

    def _softmax(self, z):
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _one_hot_encode(self, y):
        n_samples = len(y)
        y_encoded = np.zeros((n_samples, self.n_classes_))

        for i, label in enumerate(y):
            class_idx = np.where(self.classes_ == label)[0][0]
            y_encoded[i, class_idx] = 1

        return y_encoded
    
    def _calculate_loss(self, y_true, y_pred):
        n_samples = y_true.shape[0]

        # cross-entropy loss
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1-epsilon)

        cross_entropy = -np.sum(y_true * np.log(y_pred)) / n_samples

        l2_penalty = (self.regularization / 2) * np.sum(self.weights ** 2)

        return cross_entropy + l2_penalty
    
    def get_loss_history(self):
        return self.loss_history