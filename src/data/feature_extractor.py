import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import *

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def extract_mfcc_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        features = {}
        for i in range(N_MFCC):
            features[f'mfcc_mean_{i+1}'] = mfccs_mean[i]
            features[f'mfcc_std_{i+1}'] = mfccs_std[i]
            
        return features

    except Exception as e:
        print(f"Error in processing {file_path.name}: {e}")
        return None
    
def process_dataset(dataset_path, output_path):
    print(f"\nDataset path: {dataset_path}")
    print(f"Output path: {output_path}\n")

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at '{dataset_path}'")
    
    all_features = []
    
    genre_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    print(f"Found {len(genre_dirs)} genres: {', '.join(sorted([d.name for d in genre_dirs]))}\n")
    
    # Process each genre directory
    for genre_dir in sorted(genre_dirs):
        genre_name = genre_dir.name
        print(f"Processing genre: {genre_name}...")
        
        audio_files = list(genre_dir.glob('*.wav'))
        
        for audio_file in tqdm(audio_files, desc=f"  -> {genre_name}", leave=False):
            features = extract_mfcc_features(audio_file)
            if features:
                features['genre'] = genre_name
                features['filename'] = audio_file.name
                all_features.append(features)
    
    if not all_features:
        print("No features were extracted.")
        return
        
    df = pd.DataFrame(all_features)
    
    cols = ['filename', 'genre'] + [col for col in df.columns if col not in ['filename', 'genre']]
    df = df[cols]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nFeature extraction complete")
    print(f"    Total samples processed: {len(df)}")
    print(f"    Total features extracted: {len(df.columns) - 2}") 
    print(f"    Data saved to: {output_path}")
 
def main():
    process_dataset(dataset_path=RAW_DATA_DIR, output_path=OUTPUT_CSV_DIR)

if __name__ == "__main__":
    main()