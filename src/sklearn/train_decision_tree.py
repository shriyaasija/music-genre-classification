import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.data.data_loader import load_and_prepare_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
import pandas as pd
import os

def train_decision_tree():
    data = load_and_prepare_data()
    if not data: return
    X_train, X_test, y_train, y_test, label_encoder = data

    print("\n--- Training and Tuning Decision Tree ---")
    param_grid = {'max_depth': [10, 20, 30, None], 'min_samples_leaf': [1, 2, 4]}
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    y_pred_test = best_model.predict(X_test)

    report = classification_report(y_test, y_pred_test, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))
    
    return best_model, report, label_encoder, X_test, y_test