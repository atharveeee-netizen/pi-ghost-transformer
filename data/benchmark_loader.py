"""
Unified Benchmark Dataloader for Cross-Domain Generalization.
Loads TILDA 400, DAGM 2007, and MVTec-AD (Carpet/Grid/Leather/Tile).
"""

import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class TILDADataset(Dataset):
    """
    TILDA 400 Textile Dataset (64x64 patches).
    Folders: good (0), hole (1), objects (1), oil spot (1), thread error (1).
    """
    def __init__(self, root_dir: str, image_size: int = 64, split: str = "train", split_ratio: float = 0.8):
        self.image_size = image_size
        self.samples = []
        
        defect_folders = ["hole", "objects", "oil spot", "thread error"]
        
        # Load good samples (Class 0)
        good_dir = os.path.join(root_dir, "good")
        if os.path.exists(good_dir):
            for f in glob.glob(os.path.join(good_dir, "*.png")) + glob.glob(os.path.join(good_dir, "*.tif*")):
                self.samples.append((f, 0))
                
        # Load defect samples (Class 1)
        for d in defect_folders:
            d_dir = os.path.join(root_dir, d)
            if os.path.exists(d_dir):
                for f in glob.glob(os.path.join(d_dir, "*.png")) + glob.glob(os.path.join(d_dir, "*.tif*")):
                    self.samples.append((f, 1))
                    
        # Deterministic train/test split
        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(self.samples) * split_ratio)
        
        if split == "train":
            selected_idx = indices[:split_idx]
        else:
            selected_idx = indices[split_idx:]
            
        self.samples = [self.samples[i] for i in selected_idx]

        if len(self.samples) == 0:
            raise ValueError(f"No valid image files found for TILDA in '{root_dir}' under split '{split}'")

    def __len__(self):
        return len(self.samples)

    def get_dataset_mean(self, max_samples: int = 500) -> float:
        """Computes and returns the mean pixel value across the dataset split."""
        n_samples = min(len(self.samples), max_samples)
        vals = []
        for i in range(n_samples):
            img, _ = self.__getitem__(i)
            vals.append(img.mean().item())
        return float(np.mean(vals))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image from path '{path}'. File may be missing or corrupted.")
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0

        return torch.tensor(img).unsqueeze(0).float(), torch.tensor(label).long()


class DAGMDataset(Dataset):
    """
    DAGM 2007 Textured Surface Defect Dataset.
    """
    def __init__(self, root_dir: str, image_size: int = 64, split: str = "train"):
        self.image_size = image_size
        self.samples = []
        
        target_dir = os.path.join(root_dir, "Class10", "Train" if split == "train" else "Test")
        if not os.path.exists(target_dir):
            target_dir = root_dir  # fallback
            
        label_dir = os.path.join(target_dir, "Label")
        for f in glob.glob(os.path.join(target_dir, "*.PNG")) + glob.glob(os.path.join(target_dir, "*.png")):
            base = os.path.basename(f)
            # Defect image if corresponding label mask exists in Label folder
            is_defect = 0
            if os.path.exists(os.path.join(label_dir, f"{os.path.splitext(base)[0]}_label.PNG")):
                is_defect = 1
            self.samples.append((f, is_defect))

        if len(self.samples) == 0:
            raise ValueError(f"No valid image files found for DAGM in '{root_dir}' under split '{split}'")

    def __len__(self):
        return len(self.samples)

    def get_dataset_mean(self, max_samples: int = 500) -> float:
        """Computes and returns the mean pixel value across the dataset split."""
        n_samples = min(len(self.samples), max_samples)
        vals = []
        for i in range(n_samples):
            img, _ = self.__getitem__(i)
            vals.append(img.mean().item())
        return float(np.mean(vals))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image from path '{path}'. File may be missing or corrupted.")
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0

        return torch.tensor(img).unsqueeze(0).float(), torch.tensor(label).long()


class MVTecTextureDataset(Dataset):
    """
    MVTec-AD Texture Anomaly Dataset (Carpet, Grid, Leather, Tile).
    """
    def __init__(self, root_dir: str, category: str = "carpet", image_size: int = 64, split: str = "train"):
        self.image_size = image_size
        self.samples = []
        
        cat_dir = os.path.join(root_dir, category)
        if split == "train":
            good_train = os.path.join(cat_dir, "train", "good")
            if os.path.exists(good_train):
                for f in glob.glob(os.path.join(good_train, "*.png")):
                    self.samples.append((f, 0))
        else:
            test_dir = os.path.join(cat_dir, "test")
            if os.path.exists(test_dir):
                for sub in os.listdir(test_dir):
                    sub_path = os.path.join(test_dir, sub)
                    if os.path.isdir(sub_path):
                        is_defect = 0 if sub == "good" else 1
                        for f in glob.glob(os.path.join(sub_path, "*.png")):
                            self.samples.append((f, is_defect))

        if len(self.samples) == 0:
            raise ValueError(f"No valid image files found for MVTec category '{category}' split '{split}' in '{root_dir}'")

    def __len__(self):
        return len(self.samples)

    def get_dataset_mean(self, max_samples: int = 500) -> float:
        """Computes and returns the mean pixel value across the dataset split."""
        n_samples = min(len(self.samples), max_samples)
        vals = []
        for i in range(n_samples):
            img, _ = self.__getitem__(i)
            vals.append(img.mean().item())
        return float(np.mean(vals))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image from path '{path}'. File may be missing or corrupted.")
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0

        return torch.tensor(img).unsqueeze(0).float(), torch.tensor(label).long()
