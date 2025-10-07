import os
import torch
from .metrics import MetricsTracker
from tqdm import tqdm
import time

class Trainer:
    def __init__(self, model, optimizer, criterion, device, checkpoint_dir = 'checkpoints', use_amp = False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.use_amp = use_amp

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        self.best_val_acc = 0.0
        self.best_epoch = 0

        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def train_epoch(self, train_loader):
        self.model.train()
        tracker = MetricsTracker()

        pbar = tqdm(train_loader, desc='Training')

        for batch_idx, (data, labels) in enumerate(pbar):
            data = data.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()
            
            _, predictions = torch.max(outputs, dim=1)

            tracker.update(predictions, labels, loss.item())

            pbar.set_postfix({
                'loss': f'{tracker.get_average_loss():.4f}',
                'acc': f'{tracker.get_average_accuracy():.2f}%'
            })

        return {
                'loss': tracker.get_average_loss(),
                'accuracy': tracker.get_average_accuracy()
        }
    
    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        tracker = MetricsTracker()

        pbar = tqdm(val_loader, desc='Validating')

        for data, labels in pbar:
            data = data.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(data)
            loss = self.criterion(outputs, labels)

            _, predictions = torch.max(outputs, dim=1)

            tracker.update(predictions, labels, loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{tracker.get_average_loss():.4f}',
                'acc': f'{tracker.get_average_accuracy():.2f}%'
            })
        
        return {
            'loss': tracker.get_average_loss(),
            'accuracy': tracker.get_average_accuracy()
        }

    def train(self, train_loader, val_loader, epochs = 50, early_stopping_patience = 10, save_best = True):
        print("STARTING TRAINING")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print(f"Max Epochs: {epochs}")
        print(f"Early Stopping Patience: {early_stopping_patience}")

        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            print(f"\b Epoch {epoch}/{epochs}")

            train_metrics = self.train_epoch(train_loader)

            val_metrics = self.validate(val_loader)

            # Store history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])

            print(f"\nEpoch {epoch} Summary:")
            print(f"   Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"   Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.2f}%")
            
            # Check if best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch
                patience_counter = 0
                
                if save_best:
                    self.save_checkpoint(epoch, is_best=True)
                    print(f" New best model, validation accuracy: {self.best_val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
            
            # Early stopping check
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"   Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
                break
        
        # Training complete
        elapsed_time = time.time() - start_time
        print("TRAINING COMPLETE!")
        print(f"Total time: {elapsed_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        
        return self.history
    
    def save_checkpoint(self, epoch, is_best = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        # Save best model
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, path)
            print(f" Saved best model to {path}")
        
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"   Best validation accuracy: {self.best_val_acc:.2f}%")