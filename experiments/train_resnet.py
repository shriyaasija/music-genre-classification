import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.training.training import Trainer
import torch
from src.utils.visualisation import plot_training_history, plot_confusion_matrix
from src.training.metrics import MetricsTracker
import os
from src.data.loader import create_data_loaders
from src.models.se_resnet import create_se_resnet

def train_model(model_name, model, train_loader, val_loader, device, epochs = 50, lr = 0.001):
    print(f"Training {model_name}")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=f'checkpoints/{model_name}',
        use_amp = torch.cuda.is_available()
    )

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=10,
        save_best=True
    )

    plot_training_history(
        train_losses=history['train_loss'],
        val_losses=history['val_loss'],
        train_accs=history['train_acc'],
        val_accs=history['val_acc'],
        save_path=f'results/se_figures/{model_name}_training_history.png'
    )
    
    return trainer, history

def evaluate_model(model_name, model, test_loader, device, class_names):
    print(f"Evaluating {model_name} on test set")

    model.eval()
    tracker = MetricsTracker()

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            _, predictions = torch.max(outputs, dim=1)
            tracker.update(predictions, labels, loss.item())
        
    metrics = tracker.compute_final_metrics(class_names)

    print(f"\nTest Results:")
    print(f"   Loss: {metrics['loss']:.4f}")
    print(f"   Accuracy: {metrics['accuracy']:.2f}%")
    print(f"   F1 Score (Macro): {metrics['f1_macro']:.2f}%")
    print(f"   F1 Score (Weighted): {metrics['f1_weighted']:.2f}%")
    
    # Per-class accuracy
    print(f"\nPer-Genre Accuracy:")
    for idx, genre in enumerate(class_names):
        acc = metrics['per_class_accuracy'][idx]
        print(f"   {genre:12s}: {acc:.2f}%")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm=metrics['confusion_matrix'],
        classes=class_names,
        normalize=True,
        save_path=f'results/se_confusion_matrices/{model_name}_confusion_matrix.png',
        title=f'{model_name} - Confusion Matrix'
    )
    
    return metrics

def main():
    DATA_DIR = 'data/gtzan/images_original'
    FILE_EXTENSION = '.png'
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
    
    # os.makedirs('results/se_figures', exist_ok=True)
    # os.makedirs('results/se_confusion_matrices', exist_ok=True)
    # os.makedirs('checkpoints', exist_ok=True)

    print(f"Loading data from {DATA_DIR}...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=4,
        file_extension=FILE_EXTENSION
    )

    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    print("Model: SE-RESNET")

    se_resnet = create_se_resnet(num_classes=10, use_se=True)
    # se_resnet_trainer, se_resnet_history = train_model(
    #     model_name='se_resnet',
    #     model=se_resnet,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     device=device,
    #     epochs=EPOCHS,
    #     lr=LEARNING_RATE
    # )

    # se_resnet_trainer.load_checkpoint(f'checkpoints/se_resnet/best_model.pth')
    checkpoint_path = 'checkpoints/se_resnet/best_model.pth'
    print(f"Loading model from checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    se_resnet.load_state_dict(checkpoint['model_state_dict'])

    se_resnet.to(device)

    se_resnet_metrics = evaluate_model(
        model_name='se_resnet',
        model=se_resnet,
        test_loader=test_loader,
        device=device,
        class_names=genres
    )

    print(f"{'SE-ResNet':<20} {se_resnet_metrics['accuracy']:<12.2f} "
          f"{se_resnet_metrics['f1_macro']:<12.2f} {se_resnet_metrics['f1_weighted']:.2f}")
    
    results = {
        'se_resnet': se_resnet_metrics,
    }

    import json
    with open('results/final_results_seresnet.json', 'w') as f:
        json.dump({
            'se_resnet': {
                'accuracy': float(se_resnet_metrics['accuracy']),
                'f1_macro': float(se_resnet_metrics['f1_macro']),
                'f1_weighted': float(se_resnet_metrics['f1_weighted'])
            }
        }, f, indent=4)

    print(f"\n Results saved to results/final_results_seresnet.json")

if __name__ == '__main__':
    main()