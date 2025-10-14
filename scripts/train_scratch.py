import sys
from pathlib import Path
import numpy as np
import joblib
import time
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(str(Path(__file__).parent.parent))
from src.data.data_loader import load_data_with_validation
from src.scratch_models.knn import KNNClassifier
from src.scratch_models.logistic_regression import LogisticRegressionClassifier
from src.scratch_models.decision_tree import DecisionTreeClassifier
from src.training.metrics import compute_confusion_matrix, print_classification_report
from src.utils.visualisation import plot_confusion_matrix, plot_model_comparison
from src.utils.config import *


def train_knn(X_train, y_train, X_val, y_val):
    print("Training KNN from Scratch")
    
    best_score = 0
    best_model = None
    best_params = {}
    
    for n_neighbours in [3, 5, 7, 9]:
        for weights in ['uniform', 'distance']:
            print(f"\n  Testing: n_neighbours={n_neighbours}, weights={weights}")
            
            start_time = time.time()
            knn = KNNClassifier(n_neighbours=n_neighbours, weights=weights)
            knn.fit(X_train, y_train)
            
            val_score = knn.score(X_val, y_val)
            train_time = time.time() - start_time
            
            print(f"    Val Accuracy: {val_score:.4f} (Time: {train_time:.2f}s)")
            
            if val_score > best_score:
                best_score = val_score
                best_model = knn
                best_params = {'n_neighbours': n_neighbours, 'weights': weights}
    
    print(f"\nBest KNN: {best_params}")
    print(f"   Validation Accuracy: {best_score:.4f}")
    
    return best_model, best_params

def train_logistic_regression(X_train, y_train, X_val, y_val):
    print("Training Logistic Regression from Scratch")
    
    best_score = 0
    best_model = None
    best_params = {}
    
    for lr in [0.01, 0.05, 0.1]:
        for reg in [0.001, 0.01, 0.1]:
            print(f"\n  Testing: lr={lr}, regularization={reg}")
            
            start_time = time.time()
            log_reg = LogisticRegressionClassifier(
                learning_rate=lr,
                n_iterations=500,
                regularization=reg,
                verbose=False
            )
            log_reg.fit(X_train, y_train)
            
            val_score = log_reg.score(X_val, y_val)
            train_time = time.time() - start_time
            
            print(f"    Val Accuracy: {val_score:.4f} (Time: {train_time:.2f}s)")
            
            if val_score > best_score:
                best_score = val_score
                best_model = log_reg
                best_params = {'learning_rate': lr, 'regularization': reg}
    
    print(f"\nBest Logistic Regression: {best_params}")
    print(f"   Validation Accuracy: {best_score:.4f}")
    
    return best_model, best_params


def train_decision_tree(X_train, y_train, X_val, y_val):
    print("Training Decision Tree from Scratch")
    
    best_score = 0
    best_model = None
    best_params = {}
    
    for max_depth in [10, 20, 30]:
        for criterion in ['gini', 'entropy']:
            for min_samples_split in [2, 5]:
                print(f"\n  Testing: depth={max_depth}, criterion={criterion}, "
                      f"min_split={min_samples_split}")
                
                start_time = time.time()
                dt = DecisionTreeClassifier(
                    max_depth=max_depth,
                    criterion=criterion,
                    min_samples_split=min_samples_split
                )
                dt.fit(X_train, y_train)
                
                val_score = dt.score(X_val, y_val)
                train_time = time.time() - start_time
                
                print(f"    Val Acc: {val_score:.4f} "
                      f"(Depth: {dt.get_depth()}, Leaves: {dt.get_n_leaves()}, "
                      f"Time: {train_time:.2f}s)")
                
                if val_score > best_score:
                    best_score = val_score
                    best_model = dt
                    best_params = {
                        'max_depth': max_depth,
                        'criterion': criterion,
                        'min_samples_split': min_samples_split
                    }
    
    print(f"\nBest Decision Tree: {best_params}")
    print(f"   Validation Accuracy: {best_score:.4f}")
    print(f"   Tree Depth: {best_model.get_depth()}")
    print(f"   Number of Leaves: {best_model.get_n_leaves()}")
    
    return best_model, best_params

def create_comparison_table(results: dict):
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 80)
    for name, metrics in results.items():
        print(f"{name:<25} | {metrics['accuracy']:.4f}    | {metrics['precision']:.4f}    | {metrics['recall']:.4f}    | {metrics['f1']:.4f}")

def main():
    print("TRAINING SCRATCH IMPLEMENTATIONS")
    
    # Load data
    print("Loading data...")
    result = load_data_with_validation()
    
    if result is None:
        print("\nFailed to load data.")
        return
    
    X_train, X_val, X_test, y_train, y_val, y_test, loader = result
    
    # initialise storage
    models = {}
    params = {}
    results = {}
    
    # Train models
    print("\nTraining all scratch models...\n")
    
    # KNN
    knn, knn_params = train_knn(X_train, y_train, X_val, y_val)
    models['KNN'] = knn
    params['KNN'] = knn_params
    
    # Logistic Regression
    lr, lr_params = train_logistic_regression(X_train, y_train, X_val, y_val)
    models['Logistic Regression'] = lr
    params['Logistic Regression'] = lr_params
    
    # Decision Tree
    dt, dt_params = train_decision_tree(X_train, y_train, X_val, y_val)
    models['Decision Tree'] = dt
    params['Decision Tree'] = dt_params
    
    # Evaluate all models on test set
    print("TEST SET EVALUATION")
    
    for name, model in models.items():
        print(f"\nEvaluating {name}:")
        print(f"  Best params: {params[name]}")
        
        # Predict
        y_pred = model.predict(X_test)

        report_dict = classification_report(y_test, y_pred, target_names=GENRES, output_dict=True)
        
        macro_avg = report_dict['macro avg']
        results[name] = {
            'accuracy': report_dict['accuracy'],
            'precision': macro_avg['precision'],
            'recall': macro_avg['recall'],
            'f1': macro_avg['f1-score']
        }

        print_classification_report(y_pred, y_test, class_names=GENRES)

        cm = compute_confusion_matrix(y_pred, y_test)

        plot_confusion_matrix(
            cm=cm,
            classes=GENRES,
            normalize=True,
            title=f"Scratch {name} - Test Set Confusion Matrix",
            save_path=f"results/figures/scratch_{name.lower().replace(' ', '_')}_cm.png"
        )
    
        create_comparison_table(results) 

        model_names = list(results.keys())
        accuracies = [res['accuracy'] * 100 for res in results.values()] 

        plot_model_comparison(
            models=model_names,
            accuracies=accuracies,
            save_path="results/figures/scratch_models_comparison.png"
        )

    # Save models
    save_dir = MODELS_DIR / "scratch"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving models to {save_dir}...")
    for name, model in models.items():
        filename = name.lower().replace(' ', '_') + '.pkl'
        filepath = save_dir / filename
        joblib.dump({
            'model': model,
            'params': params[name],
            'metrics': results[name]
        }, filepath)
        print(f"  Saved: {filename}")
    
    # Summary
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    worst_model = min(results.items(), key=lambda x: x[1]['accuracy'])
    
    print(f"\nBest Model: {best_model[0]}")
    print(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
    print(f"   Parameters: {params[best_model[0]]}")

if __name__ == "__main__":
    main()