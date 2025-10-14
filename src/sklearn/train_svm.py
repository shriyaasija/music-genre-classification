from data_loader import load_and_prepare_data
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    data = load_and_prepare_data()
    if not data: return
    X_train, X_test, y_train, y_test, class_names, scaler = data

    print("\n--- Training and Tuning SVM ---")
    param_grid = {'C': [1, 10, 100], 'gamma': [0.01, 0.001], 'kernel': ['rbf']}
    grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred_test = best_model.predict(X_test)
    print("\nClassification Report (SVM):")
    print(classification_report(y_test, y_pred_test, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred_test, labels=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - SVM')
    plt.show()

    os.makedirs("model_outputs", exist_ok=True)
    joblib.dump(best_model, 'model_outputs/svm_model.joblib')
    joblib.dump(scaler, 'model_outputs/scaler.joblib')
    print("Tuned SVM model and scaler saved successfully.")

if __name__ == "__main__": main()
