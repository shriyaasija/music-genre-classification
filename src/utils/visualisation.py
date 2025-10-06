import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import numpy as np
import librosa.display

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_training_history(
        train_losses: List[float],
        val_losses: List[float],
        train_accs: List[float],
        val_accs: List[float],
        save_path: str = None
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(train_losses) + 1)

    # plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # plot accuracies 
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    
    plt.show()

def plot_confusion_matrix(
        cm: np.ndarray,
        classes: List[str],
        normalize: bool = False,
        save_path: str = None,
        title: str = 'Confusion Matrix'
):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Percentage' if normalize else 'Count'},
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()

def plot_per_genre_accuracy(
        accuracies: Dict[str, float],
        save_path: str = None
):
    genres = list(accuracies.keys())
    accs = list(accuracies.values())

    sorted_indices = np.argsort(accs)[::-1]
    genres = [genres[i] for i in sorted_indices]
    accs = [accs[i] for i in sorted_indices]

    colors = plt.cm.RdYlGn([acc/100 for acc in accs])
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(genres, accs, color=colors, edgecolor='black', linewidth=1.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 1,
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            fontweight='bold'
        )
    
    plt.title('Per-Genre Classification Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.axhline(y=70, color='r', linestyle='--', label='Human Baseline (70%)')
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

def plot_model_comparison(
        models: List[str],
        accuracies: List[float],
        save_path: str = None
):
    plt.figure(figsize=(10, 6))
    
    colors = sns.color_palette("husl", len(models))
    bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.5,
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=12
        )
    
    plt.title('Model Comparison - Test Accuracy', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.axhline(y=70, color='r', linestyle='--', linewidth=2, label='Human Baseline')
    plt.axhline(y=82, color='orange', linestyle='--', linewidth=2, label='Stanford CNN')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved model comparison to {save_path}")
    
    plt.show()

def plot_spectrogram(
    mel_spec: np.ndarray,
    title: str = 'Mel-Spectrogram',
    sample_rate: int = 22050,
    hop_length: int = 512,
    save_path: str = None
):
    plt.figure(figsize=(10, 4))
    
    img = librosa.display.specshow(
        mel_spec,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        cmap='viridis'
    )
    
    plt.colorbar(img, format='%+2.0f dB')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spectrogram to {save_path}")
    
    plt.show()