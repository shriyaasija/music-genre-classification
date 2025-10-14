from src.data.data_loader import load_and_prepare_data
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
import os

def train_gradient_boosting():
    data = load_and_prepare_data()
    if not data: return
    X_train, X_test, y_train, y_test, label_encoder = data

    print("\n--- Training and Tuning Gradient Boosting ---")
    param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
    grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    y_pred_test = best_model.predict(X_test)
    report = classification_report(y_test, y_pred_test, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))
    
    return best_model, report, label_encoder, X_test, y_test