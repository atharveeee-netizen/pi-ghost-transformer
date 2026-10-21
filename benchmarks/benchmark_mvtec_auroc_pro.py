"""
Rigorous Industrial Defect Benchmarking on MVTec-AD:
Computes Image-Level AUROC and Pixel-Level PRO (Per-Region Overlap).

Evaluation Protocol:
  - Image-Level AUROC: Detects whether an image contains a defect based on peak anomaly score.
  - Pixel-Level PRO: Measures coverage of connected ground-truth defect regions at False Positive Rates (FPR) <= 0.30.
  - Pixel-Level AUROC: Binary classification of every pixel.
"""

import os
import sys
import glob
import cv2
import json
import argparse
import numpy as np
from scipy.ndimage import label as scipy_label
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import get_config
from physics.forward_model import GhostImagingForwardModel
from models.ic_pinot import ICPINOT


class MVTecEvalDataset(Dataset):
    """MVTec Evaluation Dataset with paired ground truth defect masks."""
    def __init__(self, root_dir: str, category: str, image_size: int = 64):
        self.image_size = image_size
        cat_dir = os.path.join(root_dir, category)
        if not os.path.exists(cat_dir):
            raise FileNotFoundError(f"[FATAL] MVTec category directory not found: {cat_dir}")
            
        test_dir = os.path.join(cat_dir, "test")
        gt_dir = os.path.join(cat_dir, "ground_truth")
        
        self.items = []
        for defect_type in sorted(os.listdir(test_dir)):
            d_path = os.path.join(test_dir, defect_type)
            if not os.path.isdir(d_path):
                continue
            is_good = (defect_type == "good")
            for img_name in sorted(os.listdir(d_path)):
                if not (img_name.endswith(".png") or img_name.endswith(".tif") or img_name.endswith(".jpg")):
                    continue
                img_path = os.path.join(d_path, img_name)
                
                # Matching mask
                if is_good:
                    mask_path = None
                    label = 0
                else:
                    base = os.path.splitext(img_name)[0]
                    mask_path = os.path.join(gt_dir, defect_type, f"{base}_mask.png")
                    if not os.path.exists(mask_path):
                        mask_path = os.path.join(gt_dir, defect_type, f"{base}.png")
                    label = 1
                self.items.append((img_path, mask_path, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.items[idx]
        
        # Load and resize image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        img = cv2.resize(img, (self.image_size, self.image_size)).astype(np.float32) / 255.0
        
        # Load and resize mask
        if mask_path is not None and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            
        return (
            torch.tensor(img).unsqueeze(0).float(),
            torch.tensor(mask).unsqueeze(0).float(),
            torch.tensor(label).long()
        )


def compute_pro_metric(anomaly_maps: np.ndarray, ground_truth_masks: np.ndarray, max_fpr: float = 0.30) -> float:
    """
    Computes Per-Region Overlap (PRO) integrated up to max_fpr.
    anomaly_maps: (N, H, W) float
    ground_truth_masks: (N, H, W) binary float
    """
    # Find all connected defect regions
    regions = []
    for i in range(len(ground_truth_masks)):
        mask = ground_truth_masks[i]
        if mask.max() == 0:
            continue
        labeled, num_features = scipy_label(mask)
        for r_id in range(1, num_features + 1):
            region_mask = (labeled == r_id)
            if region_mask.sum() > 0:
                regions.append((i, region_mask))
                
    if len(regions) == 0:
        return 1.0 # No defect regions in dataset
        
    # Normal pixel mask (background pixels across all test images)
    normal_pixels = (ground_truth_masks == 0)
    normal_anomaly = anomaly_maps[normal_pixels]
    total_normal = normal_pixels.sum()
    
    if total_normal == 0:
        return 1.0
        
    # Evaluate across fine thresholds
    min_val, max_val = anomaly_maps.min(), anomaly_maps.max()
    thresholds = np.linspace(min_val, max_val, 200)[::-1] # High to low
    
    pro_curve = []
    fpr_curve = []
    
    for th in thresholds:
        # FPR on normal pixels
        fp_count = (normal_anomaly >= th).sum()
        fpr = fp_count / float(total_normal)
        
        # PRO across connected regions
        region_overlaps = []
        for img_idx, r_mask in regions:
            pred_mask = (anomaly_maps[img_idx] >= th)
            overlap = (pred_mask & r_mask).sum() / float(r_mask.sum())
            region_overlaps.append(overlap)
            
        mean_pro = np.mean(region_overlaps)
        pro_curve.append(mean_pro)
        fpr_curve.append(fpr)
        
        if fpr >= max_fpr:
            break
            
    # Sort by FPR
    sorted_indices = np.argsort(fpr_curve)
    fpr_curve = np.array(fpr_curve)[sorted_indices]
    pro_curve = np.array(pro_curve)[sorted_indices]
    
    # Clip to max_fpr
    valid = fpr_curve <= max_fpr
    fpr_sub = np.concatenate([[0.0], fpr_curve[valid], [max_fpr]])
    pro_sub = np.concatenate([[0.0], pro_curve[valid], [pro_curve[valid][-1] if np.any(valid) else 0.0]])
    
    # Trapezoidal integration
    pro_integral = np.trapz(pro_sub, fpr_sub)
    pro_score = pro_integral / max_fpr
    return float(pro_score)


def evaluate_mvtec_category(
    category: str,
    root_dir: str = "data/benchmarks/mvtec",
    pattern_path: str = "checkpoints/trained_patterns.pt",
    checkpoint_path: str = None,
    device: str = "cuda"
) -> dict:
    if not os.path.exists(pattern_path):
        raise FileNotFoundError(f"[FATAL] Required pattern file not found: {pattern_path}")
        
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    dataset = MVTecEvalDataset(root_dir, category=category, image_size=64)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    patterns = torch.load(pattern_path, map_location=dev)
    fwd = GhostImagingForwardModel(image_size=64, num_patterns=512, pattern_type="learned").to(dev)
    fwd.measurement_matrix.H_raw.data = torch.logit(patterns.to(dev).clamp(1e-5, 1 - 1e-5))
    
    model = ICPINOT(num_patterns=512, img_size=64).to(dev)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=dev), strict=False)
    model.eval()
    
    all_anomaly_maps = []
    all_gt_masks = []
    all_labels = []
    all_image_scores = []
    
    with torch.no_grad():
        for images, masks, labels in loader:
            B = images.shape[0]
            images = images.to(dev)
            y, _ = fwd.forward_sample(images, torch.zeros(B, 11, device=dev))
            
            # Blind evidential reconstruction
            nig_outputs, logits = model(y, optical_params=None, return_evidential=True)
            recon = nig_outputs["gamma"]
            
            # Anomaly map: combination of reconstruction residual and epistemic uncertainty
            residual = (images - recon).abs()
            var_epistemic = nig_outputs["beta"] / (nig_outputs["v"] * (nig_outputs["alpha"] - 1.0).clamp(min=1e-3))
            
            anomaly_map = (residual + 0.1 * var_epistemic).squeeze(1).cpu().numpy()
            
            # Image score: top 5% worst pixels
            for b in range(B):
                am = anomaly_map[b]
                img_score = float(np.percentile(am, 95))
                all_image_scores.append(img_score)
                
            all_anomaly_maps.append(anomaly_map)
            all_gt_masks.append(masks.squeeze(1).numpy())
            all_labels.extend(labels.numpy().tolist())
            
    all_anomaly_maps = np.concatenate(all_anomaly_maps, axis=0) # (N, 64, 64)
    all_gt_masks = np.concatenate(all_gt_masks, axis=0)         # (N, 64, 64)
    all_labels = np.array(all_labels)
    all_image_scores = np.array(all_image_scores)
    
    # 1. Image-Level AUROC
    if len(np.unique(all_labels)) > 1:
        image_auroc = float(roc_auc_score(all_labels, all_image_scores))
    else:
        image_auroc = 1.0
        
    # 2. Pixel-Level AUROC
    flat_gt = all_gt_masks.flatten()
    flat_am = all_anomaly_maps.flatten()
    if len(np.unique(flat_gt)) > 1:
        pixel_auroc = float(roc_auc_score(flat_gt, flat_am))
    else:
        pixel_auroc = 1.0
        
    # 3. Pixel-Level PRO (Per-Region Overlap)
    pro_score = compute_pro_metric(all_anomaly_maps, all_gt_masks, max_fpr=0.30)
    
    results = {
        "category": category,
        "num_test_images": len(all_labels),
        "num_defective": int(all_labels.sum()),
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "pixel_pro": pro_score
    }
    print(f"[{category.upper():<12}] Image AUROC: {image_auroc*100:5.2f}% | Pixel AUROC: {pixel_auroc*100:5.2f}% | PRO: {pro_score*100:5.2f}%")
    return results


def main():
    parser = argparse.ArgumentParser(description="MVTec-AD AUROC and PRO Benchmark")
    parser.add_argument("--categories", nargs="+", default=["carpet", "grid", "leather", "tile", "wood"])
    parser.add_argument("--patterns", default="checkpoints/trained_patterns.pt")
    parser.add_argument("--output", default="results/mvtec_auroc_pro.json")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_res = {}
    for cat in args.categories:
        ckpt = f"checkpoints/ic_pinot_mvtec-{cat}_seed42.pt"
        res = evaluate_mvtec_category(cat, pattern_path=args.patterns, checkpoint_path=ckpt if os.path.exists(ckpt) else None)
        all_res[cat] = res
        
    mean_img_auroc = np.mean([r["image_auroc"] for r in all_res.values()])
    mean_pixel_auroc = np.mean([r["pixel_auroc"] for r in all_res.values()])
    mean_pro = np.mean([r["pixel_pro"] for r in all_res.values()])
    
    print("-" * 65)
    print(f"[MEAN SUMMARY ] Image AUROC: {mean_img_auroc*100:5.2f}% | Pixel AUROC: {mean_pixel_auroc*100:5.2f}% | PRO: {mean_pro*100:5.2f}%")
    all_res["MEAN"] = {
        "image_auroc": float(mean_img_auroc),
        "pixel_auroc": float(mean_pixel_auroc),
        "pixel_pro": float(mean_pro)
    }
    with open(args.output, "w") as f:
        json.dump(all_res, f, indent=4)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
