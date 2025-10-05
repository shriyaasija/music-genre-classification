from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split

class GTZANDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            segment_length: int = 3,
            sample_rate: int = 22050,
            mode: str = 'train',
            random_seed: int = 42
    ):
        self.data_dir = data_dir
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.mode = mode
        self.random_seed = random_seed

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
            
        print(f"Found {len(self.file_paths)} audio files across {len(self.genres)} genres")

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

