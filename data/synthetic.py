"""
Synthetic Data Generation for Fabric Defects.

Generates base fabric textures and applies random defects (holes, stains, thread breaks, misweaves)
for use in the simulation pipeline. This provides the ground-truth (x_clean).
"""

import torch
import numpy as np
import cv2


class SyntheticFabricGenerator:
    """
    Generates synthetic fabric textures and anomalies (defects) to use
    as ground-truth images (x_clean) in the simulation.
    """
    def __init__(self, image_size: int = 64):
        self.image_size = image_size

    def generate_base_fabric(self) -> np.ndarray:
        """
        Generates a basic woven fabric texture using high-frequency sine waves.
        """
        N = self.image_size
        x = np.linspace(0, 4 * np.pi, N)
        y = np.linspace(0, 4 * np.pi, N)
        X, Y = np.meshgrid(x, y)

        # Warp warp and weft threads
        warp = np.sin(X * 10) * 0.5 + 0.5
        weft = np.sin(Y * 10) * 0.5 + 0.5
        
        # Combine and add noise
        fabric = (warp + weft) / 2.0
        fabric += np.random.normal(0, 0.05, (N, N))
        
        # Base fabric intensity usually isn't perfectly white or black
        fabric = np.clip(fabric * 0.5 + 0.3, 0.0, 1.0)
        return fabric.astype(np.float32)

    def add_hole(self, img: np.ndarray) -> np.ndarray:
        """Adds a dark hole defect."""
        r = np.random.randint(2, 6)
        x = np.random.randint(r, self.image_size - r)
        y = np.random.randint(r, self.image_size - r)
        cv2.circle(img, (x, y), r, 0.0, -1)
        # Add some irregular edges
        img += np.random.normal(0, 0.05, img.shape)
        return np.clip(img, 0, 1)

    def add_stain(self, img: np.ndarray) -> np.ndarray:
        """Adds a darker, blurry stain defect."""
        r = np.random.randint(4, 10)
        x = np.random.randint(r, self.image_size - r)
        y = np.random.randint(r, self.image_size - r)
        mask = np.zeros_like(img)
        cv2.circle(mask, (x, y), r, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (7, 7), 2.0)
        
        # Dim the stained area
        img = img * (1.0 - 0.5 * mask)
        return np.clip(img, 0, 1)

    def add_thread_break(self, img: np.ndarray) -> np.ndarray:
        """Adds a thin, horizontal or vertical bright/dark line."""
        is_horizontal = np.random.rand() > 0.5
        length = np.random.randint(10, self.image_size // 2)
        thickness = np.random.randint(1, 3)
        intensity = np.random.choice([0.1, 0.9]) # dark or bright
        
        if is_horizontal:
            y = np.random.randint(0, self.image_size - thickness)
            x = np.random.randint(0, self.image_size - length)
            img[y:y+thickness, x:x+length] = intensity
        else:
            x = np.random.randint(0, self.image_size - thickness)
            y = np.random.randint(0, self.image_size - length)
            img[y:y+length, x:x+thickness] = intensity
            
        return np.clip(img, 0, 1)

    def add_misweave(self, img: np.ndarray) -> np.ndarray:
        """Adds a localized distortion to the weave pattern."""
        x = np.random.randint(10, self.image_size - 10)
        y = np.random.randint(10, self.image_size - 10)
        w, h = np.random.randint(5, 15), np.random.randint(5, 15)
        
        patch = img[y:y+h, x:x+w]
        # Shift/roll the patch to simulate a misweave
        patch = np.roll(patch, shift=np.random.randint(1, 4), axis=0)
        img[y:y+h, x:x+w] = patch
        return img

    def generate_batch(self, batch_size: int, defect_prob: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a batch of clean images and their corresponding class labels.
        Labels: 0 (Normal), 1 (Hole), 2 (Stain), 3 (Thread Break), 4 (Misweave)
        """
        images = []
        labels = []
        
        for _ in range(batch_size):
            img = self.generate_base_fabric()
            label = 0
            
            if np.random.rand() < defect_prob:
                defect_type = np.random.randint(1, 5)
                label = defect_type
                if defect_type == 1:
                    img = self.add_hole(img)
                elif defect_type == 2:
                    img = self.add_stain(img)
                elif defect_type == 3:
                    img = self.add_thread_break(img)
                elif defect_type == 4:
                    img = self.add_misweave(img)
                    
            images.append(img)
            labels.append(label)
            
        # Shape: (B, 1, H, W)
        images_tensor = torch.tensor(np.array(images)).unsqueeze(1).float()
        labels_tensor = torch.tensor(labels).long()
        
        return images_tensor, labels_tensor
