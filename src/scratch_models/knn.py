import numpy as np
from collections import Counter
from .base import BaseClassifier

class KNNClassifier(BaseClassifier):
    def __init__(self, n_neighbours=5, weights='uniform', metric='euclidean'):
        super().__init__()
        self.n_neighbours = n_neighbours
        self.weights = weights
        self.metric = metric
        
        # Will be set during fit
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        X, y = self._validate_input(X, y)

        self.X_train = X
        self.y_train = y

        self._initialise_classes(y)
        self.n_features_ = X.shape[1]
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        
        X = self._validate_input(X)

        predictions = np.array([self._predict_single(x) for x in X])
        return predictions

    def _predict_single(self, x):
        distances = self._calculate_distances(x)

        k_indices = np.argsort(distances)[:self.n_neighbours]

        k_nearest_labels = self.y_train[k_indices]

        if self.weights == 'uniform':
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]
        
        else:
            k_distances = distances[k_indices]

            k_distances = np.where(k_distances == 0, 1e-10, k_distances)

            weights = 1 / k_distances

            weighted_votes = {}
            for label, weight in zip(k_nearest_labels, weights):
                weighted_votes[label] = weighted_votes.get(label, 0) + weight

            return max(weighted_votes.items(), key=lambda x: x[1])[0]
        
    def _calculate_distances(self, x):
        if self.metric == 'euclidean':
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        
        elif self.metric == 'manhattan':
            distances = np.sum(np.abs(self.X_train - x), axis=1)
        
        elif self.metric == 'minkowski':
            p = 3
            distances = np.sum(np.abs(self.X_train - x) ** p, axis=1) ** (1/p)
        
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        return distances
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        
        X = self._validate_input(X)
        
        probabilities = []
        for x in X:
            # Calculate distances
            distances = self._calculate_distances(x)
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = self.y_train[k_indices]
            
            # Count votes for each class
            class_probs = np.zeros(self.n_classes_)
            
            if self.weights == 'uniform':
                for label in k_nearest_labels:
                    class_idx = np.where(self.classes_ == label)[0][0]
                    class_probs[class_idx] += 1
                class_probs /= self.n_neighbors
            else:
                k_distances = distances[k_indices]
                k_distances = np.where(k_distances == 0, 1e-10, k_distances)
                weights = 1 / k_distances
                
                for label, weight in zip(k_nearest_labels, weights):
                    class_idx = np.where(self.classes_ == label)[0][0]
                    class_probs[class_idx] += weight
                
                class_probs /= np.sum(class_probs)
            
            probabilities.append(class_probs)
        
        return np.array(probabilities)