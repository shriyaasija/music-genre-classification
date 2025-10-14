import sys
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.visualisation import plot_confusion_matrix, plot_model_comparison

from src.sklearn.train_knn import train_knn
from src.sklearn.train_logistic_regression import train_lr
from src.sklearn.train_decision_tree import train_decision_tree
from src.sklearn.train_random_forest import train_random_forest
from src.sklearn.train_svm import train_svm
from src.sklearn.train_gradient_boosting import train_gradient_boosting

def main():
    print("Starting SciKit Learn Model Training")

    training_pipeline = [
        ("KNN", train_knn),
        ("Logistic Regression", train_lr),
        ("Decision Tree", train_decision_tree),
        ("Random Forest", train_random_forest),
        ("SVM", train_svm),
        ("Gradient Boosting", train_gradient_boosting)
    ]

    all_models = {}
    all_reports = {}

    label_encoder = None
    X_test, y_test = None, None

    for name, train_func in training_pipeline:
        model, report, le, xt, yt = train_func()

        if model and report:
            all_models[name] = model
            all_reports[name] = report
            if label_encoder is None:
                label_encoder = le
                X_test, y_test = xt, yt

    print("Saving models and compiling results...")
    model_dir = Path("models/sklearn")
    model_dir.mkdir(parents=True, exist_ok=True)

    summary_data = []
    for name, model in all_models.items():
        filename = name.lower().replace(' ', '_') + '_model.joblib'
        joblib.dump(model, model_dir / filename)
        print(f"  -> Saved {name} model to {model_dir / filename}")
        report = all_reports[name]
        summary_data.append({
            'Model': name,
            'Accuracy': report['accuracy'],
            'Precision (macro avg)': report['macro avg']['precision'],
            'Recall (macro avg)': report['macro avg']['recall'],
            'F1-score (macro avg)': report['macro avg']['f1-score']
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values(by='F1-score (macro avg)', ascending=False)
    results_path = Path("results/sklearn_model_comparison.csv")
    summary_df.to_csv(results_path, index=False)
    print(f"\n  -> Saved model comparison summary to {results_path}")

    # Visualisations
    print("Generating visualisations...")

    best_model_name = summary_df.iloc[0]['Model']
    best_model = all_models[best_model_name]
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plot_confusion_matrix(
        cm,
        classes=label_encoder.classes_,
        normalize=True,
        title=f'Normalized Confusion Matrix for {best_model_name}',
        save_path=f"results/figures/{best_model_name.lower().replace(' ', '_')}_cm.png"
    )

    plot_model_comparison(
        models=summary_df['Model'].tolist(),
        accuracies=[acc * 100 for acc in summary_df['Accuracy'].tolist()],
        save_path="results/figures/sklearn_model_comparison.png"
    )

if __name__ == '__main__':
    main()