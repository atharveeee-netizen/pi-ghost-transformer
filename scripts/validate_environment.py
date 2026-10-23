"""
Environment and Integrity Validation Script.
Validates dependencies, CUDA accelerators, dataset paths, checkpoints, and model integrity.
"""

import os
import sys
import torch
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from physics.forward_model import GhostImagingForwardModel
from models.ic_pinot import ICPINOT
from data.benchmark_loader import TILDADataset, DAGMDataset, MVTecTextureDataset
from data.aitex_loader import AITEXDataset


def validate_environment():
    print("=" * 80)
    print("ENVIRONMENT AND CODEBASE INTEGRITY VALIDATION")
    print("=" * 80)
    
    # 1. System & PyTorch
    print("\n[1/5] Checking Python & Acceleration Stack:")
    print(f"  Python version:  {sys.version.split()[0]}")
    print(f"  PyTorch version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"  CUDA Available:  {cuda_avail}")
    if cuda_avail:
        print(f"  Device Name:     {torch.cuda.get_device_name(0)}")
        print(f"  Device Memory:   {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("  WARNING: Running in CPU mode. Benchmarks will run substantially slower.")

    # 2. Checkpoints & Patterns
    print("\n[2/5] Checking Physical Measurement Patterns & Checkpoints:")
    pat_path = os.path.join(ROOT_DIR, "checkpoints", "trained_patterns.pt")
    if not os.path.exists(pat_path):
        raise FileNotFoundError(f"Missing required pattern file: {pat_path}")
    patterns = torch.load(pat_path, map_location="cpu")
    print(f"  Loaded trained patterns: shape {patterns.shape}, dtype {patterns.dtype}, range [{patterns.min():.3f}, {patterns.max():.3f}]")
    if patterns.shape != (512, 4096):
        raise ValueError(f"Pattern shape mismatch: {patterns.shape} != (512, 4096)")

    expected_ckpts = [
        "ic_pinot_tilda_400_seed42.pt",
        "ic_pinot_dagm_2007_seed42.pt",
        "ic_pinot_mvtec-carpet_seed42.pt"
    ]
    for ckpt_name in expected_ckpts:
        ckpt_path = os.path.join(ROOT_DIR, "checkpoints", ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing required checkpoint: {ckpt_path}")
        print(f"  Verified checkpoint exists: {ckpt_name}")

    # 3. Datasets
    print("\n[3/5] Validating Datasets and Sample Loaders:")
    datasets = {
        "TILDA 400 (train)": lambda: TILDADataset(os.path.join(ROOT_DIR, "data", "benchmarks", "tilda"), split="train"),
        "TILDA 400 (test)": lambda: TILDADataset(os.path.join(ROOT_DIR, "data", "benchmarks", "tilda"), split="test"),
        "DAGM 2007 (test)": lambda: DAGMDataset(os.path.join(ROOT_DIR, "data", "benchmarks", "dagm"), split="test"),
        "MVTec Carpet (test)": lambda: MVTecTextureDataset(os.path.join(ROOT_DIR, "data", "benchmarks", "mvtec"), category="carpet", split="test"),
        "AITEX (all)": lambda: AITEXDataset(os.path.join(ROOT_DIR, "data", "aitex"))
    }
    for name, loader_fn in datasets.items():
        try:
            ds = loader_fn()
            sample_img, label = ds[0]
            print(f"  {name:<25}: {len(ds)} samples | sample shape: {sample_img.shape} | mean: {float(sample_img.mean()):.3f}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {name}: {e}")

    # 4. Forward Model Execution
    print("\n[4/5] Testing Physical Forward Model Execution:")
    dev = torch.device("cuda" if cuda_avail else "cpu")
    fwd = GhostImagingForwardModel(image_size=64, num_patterns=512, pattern_type="learned").to(dev)
    fwd.measurement_matrix.H_raw.data = torch.logit(patterns.to(dev).clamp(1e-5, 1 - 1e-5))
    dummy_x = torch.rand(2, 1, 64, 64, device=dev)
    dummy_opt = torch.zeros(2, 11, device=dev)
    y, y_exp = fwd.forward_sample(dummy_x, dummy_opt)
    print(f"  Forward sample output y shape: {y.shape} | min: {y.min():.2f} | max: {y.max():.2f}")
    assert y.shape == (2, 512), f"Measurement shape mismatch: {y.shape}"

    # 5. Model Inference & Evidential Head
    print("\n[5/5] Testing IC-PINOT Reconstruction & Evidential Outputs:")
    model = ICPINOT(num_patterns=512, img_size=64).to(dev)
    model.eval()
    with torch.no_grad():
        nig_out, logits = model(y, optical_params=dummy_opt, return_evidential=True)
        rec = nig_out["gamma"]
    print(f"  Reconstructed shape: {rec.shape} | Logits shape: {logits.shape}")
    print(f"  Evidential keys: {list(nig_out.keys())}")
    assert rec.shape == (2, 1, 64, 64)
    assert "gamma" in nig_out and "decoder_mean" in nig_out and "backprojection" in nig_out

    print("\n" + "=" * 80)
    print("ALL ENVIRONMENT AND REPOSITORY INTEGRITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    validate_environment()
