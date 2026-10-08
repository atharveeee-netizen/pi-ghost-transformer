import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class AITEXDataset(Dataset):
    def __init__(self, root_dir, image_size=64, split="train"):
        self.root_dir = root_dir
        self.image_size = image_size
        self.split = split
        
        # All image files (excluding masks)
        all_paths = []
        for file in glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True):
            if "_mask" not in file and "Mask_images" not in file:
                all_paths.append(file)
                
        # Get unique fabrics (directories inside Defect_images or NODefect_images)
        fabrics = sorted(list(set([os.path.basename(os.path.dirname(p)) for p in all_paths])))
        
        # Split: Leave the last 2 fabrics for testing (Out of Distribution testing)
        if split == "train":
            allowed_fabrics = fabrics[:-2]
        else:
            allowed_fabrics = fabrics[-2:]
            
        self.image_paths = [p for p in all_paths if os.path.basename(os.path.dirname(p)) in allowed_fabrics]
                
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load grayscale image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        else:
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = image.astype(np.float32) / 255.0
            
        # Determine label (simplified: 0 for normal, 1 for defect if mask exists)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # In AITEX, masks are typically named <base_name>_mask.png and stored in Mask_images
        mask_path1 = os.path.join(self.root_dir, "Mask_images", f"{base_name}_mask.png")
        mask_path2 = os.path.join(self.root_dir, "Mask_images", f"{base_name}_mask1.png")
        
        label = 0
        if os.path.exists(mask_path1) or os.path.exists(mask_path2) or "Defect_images" in img_path:
            label = 1 # Defect present
            
        img_tensor = torch.tensor(image).unsqueeze(0).float()
        label_tensor = torch.tensor(label).long()
        
        return img_tensor, label_tensor
