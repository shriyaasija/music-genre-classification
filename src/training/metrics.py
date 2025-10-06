import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report

def calculate_accuracy(predictions: torch.Tensor, labels: torch.Tensor):
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = (correct / total) * 100
    return accuracy

def calculate_per_class_accuracy(predictions: np.ndarray, labels: np.ndarray, num_classes: int = 10):
    per_class_acc = {}

    for idx in range(num_classes):
        class_mask = (labels == idx)

        if class_mask.sum() == 0:
            per_class_acc[idx] = 0.0
            continue

        class_predictions = predictions[class_mask]
        class_labels = labels[class_mask]
        correct = (class_predictions == class_labels).sum()

        per_class_acc[idx] = (correct / len(class_labels)) * 100
    
    return per_class_acc

def compute_confusion_matrix(predictions: np.ndarray, labels: np.ndarray, num_classes: int = 10):
    cm = confusion_matrix(labels, predictions, labels=range(num_classes))
    return cm

def calculate_f1_score(
        predictions: np.ndarray, 
        labels: np.ndarray,
        average: str = 'macro'
):
    f1 = f1_score(labels, predictions, average=average, zero_division=0)
    return f1

def print_classification_report(
        predictions: np.ndarray,
        labels: np.ndarray, 
        class_names: list
):
    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=3,
        zero_division=0
    )

    print("Classification Report: \n")
    print(report)

class MetricsTracker:
    """Helper class to track metrics during training"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.total_correct = 0
        self.total_samples = 0
        self.all_predictions = []
        self.all_labels = []

    def update(self, predictions, labels, loss):
        """Update metrics within a batch of data"""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        batch_size = len(labels)
        correct = (predictions == labels).sum()

        self.total_loss += loss * batch_size
        self.total_correct += correct
        self.total_samples += batch_size

        self.all_predictions.extend(predictions)
        self.all_labels.extend(labels)   

    def get_average_loss(self):
        """Get average loss across all batches."""
        return self.total_loss / self.total_samples if self.total_samples > 0 else 0.0
    
    def get_average_accuracy(self):
        """Get average accuracy across all batches."""
        return (self.total_correct / self.total_samples * 100) if self.total_samples > 0 else 0.0
    
    def get_all_predictions(self):
        """Get all predictions and labels."""
        return np.array(self.all_predictions), np.array(self.all_labels)
    
    def compute_final_metrics(self, class_names: list = None):
        """Compute all metrics at once."""
        
        preds, labels = self.get_all_predictions()
        
        metrics = {
            'loss': self.get_average_loss(),
            'accuracy': self.get_average_accuracy(),
            'f1_macro': calculate_f1_score(preds, labels, average='macro') * 100,
            'f1_weighted': calculate_f1_score(preds, labels, average='weighted') * 100,
            'confusion_matrix': compute_confusion_matrix(preds, labels),
            'per_class_accuracy': calculate_per_class_accuracy(preds, labels)
        }
        
        return metrics
 