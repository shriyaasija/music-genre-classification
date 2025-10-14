import numpy as np
from collections import Counter
from .base import BaseClassifier

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, samples=None, impurity=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.samples = samples
        self.impurity = impurity
    
    def is_leaf(self):
        return self.value is not None
    
class DecisionTreeClassifier(BaseClassifier):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='gini'):
        super().__init__()

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        
        self.root = None
        self.n_features_ = None
    
    def fit(self, X, y):
        X, y = self._validate_input(X, y)
        self._initialise_classes(y)
        self.n_features_ = X.shape[1]

        self.root = self._build_tree(X, y, depth=0)

        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._validate_input(X)

        predictions = np.array([self._traverse_tree(x, self.root) for x in X])
        return predictions
    
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        impurity = self._calculate_impurity(y)

        if (depth >= self.max_depth if self.max_depth else False) or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, samples=n_samples, impurity=impurity)
        
        best_feature, best_threshold = self._find_best_split(X, y)

        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, samples=n_samples, impurity=impurity)
        
        # Split data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        
        # Check min_samples_leaf
        if np.sum(left_indices) < self.min_samples_leaf or \
           np.sum(right_indices) < self.min_samples_leaf:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, samples=n_samples, impurity=impurity)
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            samples=n_samples,
            impurity=impurity
        )

    def _find_best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        parent_impurity = self._calculate_impurity(y)

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                gain = self._information_gain(y, y[left_mask], y[right_mask])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _information_gain(self, parent, left_child, right_child):
        n = len(parent)
        n_left = len(left_child)
        n_right = len(right_child)
        
        # Information gain = parent impurity - weighted child impurities
        parent_impurity = self._calculate_impurity(parent)
        left_impurity = self._calculate_impurity(left_child)
        right_impurity = self._calculate_impurity(right_child)
        
        child_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        
        return parent_impurity - child_impurity
    
    def _calculate_impurity(self, y):
        if len(y) == 0:
            return 0
        
        # Calculate class probabilities
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        if self.criterion == 'gini':
            # Gini impurity: 1 - Σ(p_i²)
            return 1 - np.sum(probabilities ** 2)
        
        elif self.criterion == 'entropy':
            # Entropy: -Σ(p_i * log2(p_i))
            return -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def get_depth(self):
        def _get_depth(node):
            if node is None or node.is_leaf():
                return 0
            return 1 + max(_get_depth(node.left), _get_depth(node.right))
        
        return _get_depth(self.root)
    
    def get_n_leaves(self):
        def _count_leaves(node):
            if node is None:
                return 0
            if node.is_leaf():
                return 1
            return _count_leaves(node.left) + _count_leaves(node.right)
        
        return _count_leaves(self.root)

    