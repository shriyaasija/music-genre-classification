from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
from PIL import Image
import torch

class GTZANDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            mode: str = 'train',
            file_extension: str = '.png',
            random_seed: int = 42,
            img_size = (432, 288)
    ):
        self.data_dir = data_dir
        self.mode = mode
        self.file_extension = file_extension
        self.random_seed = random_seed
        self.img_size = img_size

        self.genres = [
            'blues', 'classical', 'country', 'disco', 'hiphop',
            'jazz', 'metal', 'pop', 'reggae', 'rock'
        ]

        self.genre_to_label = {genre: idx for idx, genre in enumerate(self.genres)}
        self.label_to_genre = {idx: genre for genre, idx in self.genre_to_label.items()}

        self.file_paths = []
        self.labels = []
        self._load_file_paths()

        self._create_splits()

    def _load_file_paths(self):
        print(f"Loading file paths from {self.data_dir}...")

        for genre in self.genres:
            genre_dir = os.path.join(self.data_dir, genre)

            if not os.path.exists(genre_dir):
                print(f"Warning: {genre_dir} not found, skipping...")
                continue

            files = [f for f in os.listdir(genre_dir)
                     if f.endswith(('.au', '.wav'))]
        
            for file in files:
                file_path = os.path.join(genre_dir, file)
                self.file_paths.append(file_path)
                self.labels.append(self.genre_to_label[genre])
            
        print(f"Found {len(self.file_paths)} files across {len(self.genres)} genres")

    def _create_splits(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        train_files, temp_files, train_labels, temp_labels = train_test_split(
            self.file_paths,
            self.labels,
            test_size=0.2,
            stratify=self.labels,
            random_state=self.random_seed
        )

        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files,
            temp_labels,
            test_size=0.5,
            stratify=temp_labels,
            random_state=self.random_seed
        )

        if self.mode == 'train':
            self.file_paths = train_files
            self.labels = train_labels
        elif self.mode == 'test':
            self.file_paths = test_files
            self.labels = test_labels
        elif self.mode == 'val':
            self.file_paths = val_files
            self.labels = val_labels
    
        print(f"Split {self.mode} has {len(self.file_paths)} samples")
    
    def __len__(self) -> int:
        return len(self.file_paths)

    def _load_image(self, file_path):
        img = Image.open(file_path).convert('L')
        spec = np.array(img)

        # if we need to normalise
        # if spec.max() > 1.0:
        #     spec = spec / 255.0

        return spec.astype(np.float32)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        label = self.labels[index]

        spec = self._load_image(file_path)
        spec = np.expand_dims(spec, axis = 0)

        spec_tensor = torch.from_numpy(spec).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return spec_tensor, label_tensor
    
def create_data_loaders(
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        segment_length: int = 3,
        sample_rate: int = 22050
):
    train_dataset = GTZANDataset(
        data_dir=data_dir,
        segment_length=segment_length,
        sample_rate=sample_rate,
        mode='train'
    )

    test_dataset = GTZANDataset(
        data_dir=data_dir,
        segment_length=segment_length,
        sample_rate=sample_rate,
        mode='test'
    )

    val_dataset = GTZANDataset(
        data_dir=data_dir,
        segment_length=segment_length,
        sample_rate=sample_rate,
        mode='val'
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print("Dataloader created")
    print(f"Test size: {len(test_loader)}")
    print(f"Train size: {len(train_loader)}")
    print(f"Val loader: {len(val_loader)}")

    return train_loader, test_loader, val_loader