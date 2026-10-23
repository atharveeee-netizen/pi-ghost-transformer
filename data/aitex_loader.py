"""
AITEX Textile Defect Dataset Loader.
Supports stratified train/test splits, full dataset loading, and Leave-One-Fabric-Out evaluation.
"""

import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class AITEXDataset(Dataset):
    """
    AITEX Fabric Defect Database.
    Contains 247 total images: 141 defect-free images across 7 fabrics and 106 defective fabric images.
    """
    def __init__(self, root_dir: str, image_size: int = 64, split: str = "all", split_ratio: float = 0.8):
        self.root_dir = root_dir
        self.image_size = image_size
        self.split = split
        
        # Discover all valid image files (excluding segmentation masks)
        normal_paths = []
        defect_paths = []
        
        for file in glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True):
            if "_mask" in file or "Mask_images" in file:
                continue
            if "NODefect_images" in file:
                normal_paths.append((file, 0))
            elif "Defect_images" in file:
                defect_paths.append((file, 1))

        if len(normal_paths) == 0 and len(defect_paths) == 0:
            raise FileNotFoundError(f"No valid AITEX images found in '{root_dir}'")

        # Deterministic stratified split
        np.random.seed(42)
        perm_normal = np.random.permutation(len(normal_paths))
        perm_defect = np.random.permutation(len(defect_paths))

        split_n = int(len(normal_paths) * split_ratio)
        split_d = int(len(defect_paths) * split_ratio)

        if split == "train":
            self.samples = [normal_paths[i] for i in perm_normal[:split_n]] + \
                           [defect_paths[i] for i in perm_defect[:split_d]]
        elif split == "test":
            self.samples = [normal_paths[i] for i in perm_normal[split_n:]] + \
                           [defect_paths[i] for i in perm_defect[split_d:]]
        else: # "all"
            self.samples = normal_paths + defect_paths

        # Shuffle deterministically
        np.random.seed(42)
        idx_order = np.random.permutation(len(self.samples))
        self.samples = [self.samples[i] for i in idx_order]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load grayscale image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Failed to read image from path '{img_path}'. File may be missing or corrupted.")
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255.0

        img_tensor = torch.tensor(image).unsqueeze(0).float()
        label_tensor = torch.tensor(label).long()
        return img_tensor, label_tensor

    def get_dataset_mean(self, max_samples: int = 500) -> float:
        """Computes empirical mean intensity of dataset split."""
        n_samples = min(len(self.samples), max_samples)
        vals = []
        for i in range(n_samples):
            img_path, _ = self.samples[i]
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                vals.append(img.astype(np.float32).mean() / 255.0)
        return float(np.mean(vals)) if len(vals) > 0 else 0.5
