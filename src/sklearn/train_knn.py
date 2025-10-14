import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.data.data_loader import load_and_prepare_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def train_knn():
    data = load_and_prepare_data()
    if not data: return
    X_train, X_test, y_train, y_test, label_encoder = data

    print("\n--- Training and Tuning K-Nearest Neighbors ---")
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred_test = best_model.predict(X_test)
    report = classification_report(y_test, y_pred_test, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

    return best_model, report, label_encoder, X_test, y_test