"""
Comprehensive Evaluation of Uncertainty Calibration, Measurement Consistency Metrics,
and Selective Inspection Policies at Matched Coverage.
"""

import os
import sys
import json
import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from physics.forward_model import GhostImagingForwardModel
from models.ic_pinot import ICPINOT
from data.benchmark_loader import TILDADataset
from data.aitex_loader import AITEXDataset
from utils.metrics import calculate_psnr, calculate_ssim


def compute_metrics_and_residuals(model, fwd, images, opt_shift=None):
    device = images.device
    B = images.shape[0]
    if opt_shift is None:
        opt_shift = torch.zeros(B, 11, device=device)

    # Observed measurement under optical state
    y_meas, _ = fwd.forward_sample(images, opt_shift)

    # Blind reconstruction
    nig_out, logits = model(y_meas, optical_params=None, return_evidential=True)
    rec = nig_out["gamma"]
    unc_tot = nig_out["total_uncertainty"].mean(dim=(-1, -2, -3))
    unc_epi = nig_out["epistemic"].mean(dim=(-1, -2, -3))
    unc_ale = nig_out["aleatoric"].mean(dim=(-1, -2, -3))

    # Re-projection under nominal model (opt = 0)
    exp_counts = fwd.forward_expected(rec, torch.zeros(B, 11, device=device))
    scale = fwd.mean_photon_count + 1e-8
    exp_meas_norm = exp_counts / scale
    sigma_read = fwd.noise_model.readout_noise_std
    var_norm = (exp_counts + sigma_read**2) / (scale**2)
    std_norm = torch.sqrt(var_norm.clamp(min=1e-8))

    diff = y_meas - exp_meas_norm
    s1_l1 = (diff.abs() / std_norm).mean(dim=1)
    s2_l2 = torch.sqrt(((diff / std_norm)**2).mean(dim=1))
    s3_nll = (exp_meas_norm - y_meas * torch.log(exp_meas_norm.clamp(min=1e-6))).mean(dim=1)
    s4_raw = torch.norm(diff, dim=1) / np.sqrt(y_meas.shape[1])

    # True reconstruction errors
    mae = torch.abs(images - rec).mean(dim=(-1, -2, -3))
    mse = ((images - rec)**2).mean(dim=(-1, -2, -3))
    psnr = torch.tensor([calculate_psnr(images[b:b+1], rec[b:b+1]) for b in range(B)], device=device)

    return {
        "rec": rec,
        "logits": logits,
        "psnr": psnr,
        "mae": mae,
        "mse": mse,
        "s1_l1": s1_l1,
        "s2_l2": s2_l2,
        "s3_nll": s3_nll,
        "s4_raw": s4_raw,
        "unc_tot": unc_tot,
        "unc_epi": unc_epi,
        "unc_ale": unc_ale
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing calibration and policy benchmark on: {device}")

    patterns = torch.load(os.path.join(ROOT_DIR, "checkpoints", "trained_patterns.pt"), map_location=device)
    fwd = GhostImagingForwardModel(image_size=64, num_patterns=512, pattern_type="learned").to(device)
    fwd.measurement_matrix.H_raw.data = torch.logit(patterns.clamp(1e-5, 1 - 1e-5))

    model = ICPINOT(num_patterns=512, img_size=64).to(device)
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "ic_pinot_tilda_400_seed42.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    model.eval()

    # Load 300 test samples from TILDA
    ds_test = TILDADataset(os.path.join(ROOT_DIR, "data", "benchmarks", "tilda"), split="test")
    sub_test = Subset(ds_test, range(min(300, len(ds_test))))
    loader = DataLoader(sub_test, batch_size=25, shuffle=False)

    # 1. Evaluate under Nominal (a4 = 0) and Defocus Shift (a4 = 2.0)
    all_data = {
        "mae": [], "mse": [], "psnr": [],
        "s1_l1": [], "s2_l2": [], "s3_nll": [], "s4_raw": [],
        "unc_tot": [], "unc_epi": [], "unc_ale": [],
        "shift_flag": [] # 0 = nominal, 1 = shifted
    }

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            B = imgs.shape[0]

            # Nominal pass
            res_nom = compute_metrics_and_residuals(model, fwd, imgs, opt_shift=None)
            for k in ["mae", "mse", "psnr", "s1_l1", "s2_l2", "s3_nll", "s4_raw", "unc_tot", "unc_epi", "unc_ale"]:
                all_data[k].extend(res_nom[k].cpu().numpy().tolist())
            all_data["shift_flag"].extend([0] * B)

            # Shifted pass (a4 = 2.0 defocus)
            opt_shift = torch.zeros(B, 11, device=device)
            opt_shift[:, 4] = 2.0
            res_shift = compute_metrics_and_residuals(model, fwd, imgs, opt_shift=opt_shift)
            for k in ["mae", "mse", "psnr", "s1_l1", "s2_l2", "s3_nll", "s4_raw", "unc_tot", "unc_epi", "unc_ale"]:
                all_data[k].extend(res_shift[k].cpu().numpy().tolist())
            all_data["shift_flag"].extend([1] * B)

    for k in all_data:
        all_data[k] = np.array(all_data[k])

    # 2. Correlation Analysis
    print("\n" + "=" * 80)
    print("PART 1: CORRELATION ANALYSIS WITH TRUE RECONSTRUCTION ERROR (MAE / MSE)")
    print("=" * 80)
    print(f"{'Uncertainty / Residual Metric':<30} | {'Pearson r (MAE)':<16} | {'Spearman rho (MAE)':<18} | {'Shift Sensitivity (Diff)'}")
    print("-" * 80)

    nom_idx = (all_data["shift_flag"] == 0)
    shift_idx = (all_data["shift_flag"] == 1)

    signals = ["s1_l1", "s2_l2", "s3_nll", "s4_raw", "unc_tot", "unc_epi", "unc_ale"]
    labels = [
        "S1: Standardized L1 Residual",
        "S2: Standardized L2 Residual",
        "S3: Poisson NLL Residual",
        "S4: Raw Unnormalized Norm",
        "U_tot: Evidential Total Unc",
        "U_epi: Evidential Epistemic",
        "U_ale: Evidential Aleatoric"
    ]

    corr_results = {}
    for sig, lbl in zip(signals, labels):
        vals = all_data[sig]
        pr, _ = pearsonr(vals, all_data["mae"])
        sr, _ = spearmanr(vals, all_data["mae"])
        diff_shift = float(np.mean(vals[shift_idx]) - np.mean(vals[nom_idx]))
        corr_results[sig] = {
            "name": lbl,
            "pearson_r": float(pr),
            "spearman_rho": float(sr),
            "shift_delta": diff_shift
        }
        print(f"{lbl:<30} | {pr:14.4f}  | {sr:16.4f}   | {diff_shift:+12.4f}")

    # 3. Policy Benchmark at Matched Coverage
    print("\n" + "=" * 80)
    print("PART 2: SELECTIVE INSPECTION POLICY COMPARISON AT MATCHED COVERAGE")
    print("=" * 80)

    # Standardize signals
    z_s2 = (all_data["s2_l2"] - np.mean(all_data["s2_l2"])) / (np.std(all_data["s2_l2"]) + 1e-8)
    z_unc = (all_data["unc_tot"] - np.mean(all_data["unc_tot"])) / (np.std(all_data["unc_tot"]) + 1e-8)
    z_comp = 0.6 * z_s2 + 0.4 * z_unc

    policies = {
        "Always Accept": None,
        "Random Rejection": "random",
        "Evidential Uncertainty Alone": z_unc,
        "Measurement Residual Alone (S2)": z_s2,
        "Composite Risk Policy (S2 + Unc)": z_comp
    }

    target_coverages = [0.50, 0.70, 0.85, 0.95]
    policy_results = {}

    for cov in target_coverages:
        print(f"\n--- Target Coverage: {cov*100:.0f}% ---")
        print(f"{'Policy Name':<35} | {'Accepted PSNR':<15} | {'Accepted MAE':<15} | {'Defect Shift Rejected %'}")
        print("-" * 75)
        policy_results[f"coverage_{int(cov*100)}"] = {}

        n_samples = len(all_data["mae"])
        n_keep = int(n_samples * cov)

        for pol_name, score_arr in policies.items():
            if pol_name == "Always Accept":
                keep_idx = np.arange(n_samples)
            elif pol_name == "Random Rejection":
                np.random.seed(42)
                keep_idx = np.random.choice(n_samples, size=n_keep, replace=False)
            else:
                # Lower risk score -> accept
                sorted_idx = np.argsort(score_arr)
                keep_idx = sorted_idx[:n_keep]

            acc_psnr = float(np.mean(all_data["psnr"][keep_idx]))
            acc_mae = float(np.mean(all_data["mae"][keep_idx]))
            
            # How many of the shifted samples were rejected (filtered out)?
            rejected_mask = np.ones(n_samples, dtype=bool)
            rejected_mask[keep_idx] = False
            total_shifted = shift_idx.sum()
            shifted_rejected = (rejected_mask & shift_idx).sum()
            pct_shift_rej = float(shifted_rejected / total_shifted * 100) if total_shifted > 0 else 0.0

            policy_results[f"coverage_{int(cov*100)}"][pol_name] = {
                "accepted_psnr": acc_psnr,
                "accepted_mae": acc_mae,
                "shifted_samples_rejected_pct": pct_shift_rej
            }
            print(f"{pol_name:<35} | {acc_psnr:5.2f} dB        | {acc_mae:.4f}          | {pct_shift_rej:6.1f}%")

    out_file = os.path.join(os.path.dirname(__file__), "results", "calibration_and_policies.json")
    with open(out_file, "w") as f:
        json.dump({"correlation_analysis": corr_results, "matched_coverage_policies": policy_results}, f, indent=4)
    print(f"\nFull results saved to: {out_file}\n")


if __name__ == "__main__":
    main()
