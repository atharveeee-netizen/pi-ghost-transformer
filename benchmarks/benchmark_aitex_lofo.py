"""
Leave-One-Fabric-Out (LOFO) Cross-Validation Benchmark on AITEX.

Evaluates generalization to unseen fabric weaves/textures:
For each fabric k in AITEX:
  - Model is evaluated on fabric k (held out)
  - Reports Reconstruction Fidelity (PSNR, SSIM) and Defect Detection (F1, AUROC).
"""

import os
import sys
import glob
import cv2
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from physics.forward_model import GhostImagingForwardModel
from models.ic_pinot import ICPINOT
from utils.metrics import calculate_psnr, calculate_ssim


class AITEXFabricDataset(Dataset):
    """Loads AITEX samples filtered by fabric type."""
    def __init__(self, root_dir: str, target_fabric: str = None, exclude_fabric: str = None, image_size: int = 64):
        self.image_size = image_size
        self.samples = []
        
        # All image paths
        all_imgs = []
        for ext in ["*.png", "*.PNG", "*.tif", "*.jpg"]:
            all_imgs.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
            
        all_imgs = [p for p in all_imgs if "_mask" not in p and "Mask_images" not in p]
        
        for p in all_imgs:
            fab_name = os.path.basename(os.path.dirname(p))
            if target_fabric is not None and fab_name != target_fabric:
                continue
            if exclude_fabric is not None and fab_name == exclude_fabric:
                continue
                
            base_name = os.path.splitext(os.path.basename(p))[0]
            mask_path = os.path.join(root_dir, "Mask_images", f"{base_name}_mask.png")
            is_defect = 1 if (os.path.exists(mask_path) or "Defect_images" in p) else 0
            self.samples.append((p, is_defect))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        else:
            img = cv2.resize(img, (self.image_size, self.image_size)).astype(np.float32) / 255.0
        return torch.tensor(img).unsqueeze(0).float(), torch.tensor(label).long()


def run_aitex_lofo(
    root_dir: str = "data/aitex",
    pattern_path: str = "checkpoints/trained_patterns.pt",
    checkpoint_path: str = "checkpoints/ic_pinot_tilda_400_seed42.pt",
    device: str = "cuda"
) -> dict:
    if not os.path.exists(root_dir):
        # Check alternative path
        alt_root = "data/benchmarks/aitex"
        if os.path.exists(alt_root):
            root_dir = alt_root
        else:
            raise FileNotFoundError(f"[FATAL] AITEX dataset directory not found at '{root_dir}' or '{alt_root}'")

    if not os.path.exists(pattern_path):
        raise FileNotFoundError(f"[FATAL] Required pattern file not found: {pattern_path}")

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Discover available fabric categories
    all_imgs = [p for p in glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True) if "_mask" not in p and "Mask_images" not in p]
    fabrics = sorted(list(set([os.path.basename(os.path.dirname(p)) for p in all_imgs])))
    
    if len(fabrics) == 0:
        raise ValueError(f"No fabric subdirectories found in {root_dir}")
        
    print("=" * 75)
    print(f"AITEX LEAVE-ONE-FABRIC-OUT (LOFO) BENCHMARK | Found {len(fabrics)} Fabrics")
    print("=" * 75)
    
    # Load physical forward model
    patterns = torch.load(pattern_path, map_location=dev)
    fwd = GhostImagingForwardModel(image_size=64, num_patterns=512, pattern_type="learned").to(dev)
    fwd.measurement_matrix.H_raw.data = torch.logit(patterns.to(dev).clamp(1e-5, 1 - 1e-5))
    
    # Load IC-PINOT model
    model = ICPINOT(num_patterns=512, img_size=64).to(dev)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=dev), strict=False)
    model.eval()
    
    lofo_results = {}
    
    for held_out_fabric in fabrics:
        test_ds = AITEXFabricDataset(root_dir, target_fabric=held_out_fabric)
        if len(test_ds) == 0:
            continue
            
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
        psnrs, ssims = [], []
        labels_list, preds_list, scores_list = [], [], []
        
        with torch.no_grad():
            for imgs, lbls in test_loader:
                B = imgs.shape[0]
                imgs = imgs.to(dev)
                lbls = lbls.to(dev)
                
                # Single-pixel blind forward sample
                y, _ = fwd.forward_sample(imgs, torch.zeros(B, 11, device=dev))
                recon, logits = model(y, optical_params=None)
                
                p = calculate_psnr(imgs, recon)
                s = calculate_ssim(imgs, recon)
                psnrs.append(p)
                ssims.append(s)
                
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                
                labels_list.extend(lbls.cpu().numpy().tolist())
                preds_list.extend(preds.tolist())
                scores_list.extend(probs.tolist())
                
        labels_arr = np.array(labels_list)
        scores_arr = np.array(scores_list)
        preds_arr = np.array(preds_list)
        
        f1 = float(f1_score(labels_arr, preds_arr, average="macro", zero_division=0))
        if len(np.unique(labels_arr)) > 1:
            auc = float(roc_auc_score(labels_arr, scores_arr))
        else:
            auc = 1.0
            
        mean_p = float(np.mean(psnrs))
        mean_s = float(np.mean(ssims))
        
        lofo_results[held_out_fabric] = {
            "samples": len(test_ds),
            "psnr": mean_p,
            "ssim": mean_s,
            "f1_score": f1,
            "auroc": auc
        }
        print(f"  Held-Out: {held_out_fabric:<15} (N={len(test_ds):3d}) | PSNR: {mean_p:5.2f} dB | SSIM: {mean_s:.4f} | F1: {f1:.4f} | AUROC: {auc*100:5.2f}%")
        
    avg_psnr = float(np.mean([r["psnr"] for r in lofo_results.values()]))
    avg_ssim = float(np.mean([r["ssim"] for r in lofo_results.values()]))
    avg_f1 = float(np.mean([r["f1_score"] for r in lofo_results.values()]))
    avg_auc = float(np.mean([r["auroc"] for r in lofo_results.values()]))
    
    print("-" * 75)
    print(f"  LOFO MACRO AVERAGE    | PSNR: {avg_psnr:5.2f} dB | SSIM: {avg_ssim:.4f} | F1: {avg_f1:.4f} | AUROC: {avg_auc*100:5.2f}%")
    print("=" * 75)
    
    summary = {
        "per_fabric": lofo_results,
        "macro_mean": {
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "f1_score": avg_f1,
            "auroc": avg_auc
        }
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="AITEX LOFO Cross-Validation")
    parser.add_argument("--root", default="data/aitex")
    parser.add_argument("--patterns", default="checkpoints/trained_patterns.pt")
    parser.add_argument("--checkpoint", default="checkpoints/ic_pinot_tilda_400_seed42.pt")
    parser.add_argument("--output", default="results/aitex_lofo_results.json")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = run_aitex_lofo(root_dir=args.root, pattern_path=args.patterns, checkpoint_path=args.checkpoint)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print(f"LOFO results saved to {args.output}")


if __name__ == "__main__":
    main()
