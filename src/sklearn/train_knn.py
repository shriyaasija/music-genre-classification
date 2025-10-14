from data_loader import load_and_prepare_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    data = load_and_prepare_data()
    if not data: return
    X_train, X_test, y_train, y_test, class_names, _ = data

    print("\n--- Training and Tuning K-Nearest Neighbors ---")
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred_test = best_model.predict(X_test)
    print("\nClassification Report (K-Nearest Neighbors):")
    print(classification_report(y_test, y_pred_test, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred_test, labels=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - K-Nearest Neighbors')
    plt.show()

    os.makedirs("model_outputs", exist_ok=True)
    joblib.dump(best_model, 'model_outputs/knn_model.joblib')
    print("Tuned K-Nearest Neighbors model saved successfully.")

if __name__ == "__main__": main()
