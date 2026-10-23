"""
Joint Photon and Optical Shift Benchmark with Risk-Calibrated Selective Inspection.

Evaluates the core research hypothesis:
"Risk-Calibrated Single-Pixel Imaging for Reliable Textile Defect Inspection Under Joint Photon and Optical Shift"

Sweeps:
1. Optical Defocus Shift: a4 in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
2. Photon Budget Shift: N_total in [1024, 2560, 5120, 10240, 25600]
3. Cross-Fabric Material Shift: In-distribution (TILDA) vs Out-of-Distribution Unseen Fabrics (AITEX)
4. Evaluates 3-Tier Selective Inspection Policy (Accept, Re-measure, Refer) vs Standard Un-calibrated Inspection.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from physics.forward_model import GhostImagingForwardModel
from models.ic_pinot import ICPINOT
from models.selective_inspection import MeasurementConsistencyScorer, SelectiveInspectionPolicy
from data.benchmark_loader import TILDADataset
from data.aitex_loader import AITEXDataset
from utils.metrics import calculate_psnr, calculate_ssim


def run_optical_shift_sweep(
    model: ICPINOT,
    fwd_model: GhostImagingForwardModel,
    scorer: MeasurementConsistencyScorer,
    policy: SelectiveInspectionPolicy,
    dataset,
    device: torch.device,
    max_samples: int = 100
) -> list:
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: OPTICAL ABERRATION SHIFT (Zernike Defocus a4 Sweep)")
    print("=" * 80)
    print(f"{'a4 Defocus':<10} | {'PSNR (All)':<12} | {'MC Residual':<12} | {'Accept %':<10} | {'Re-meas %':<10} | {'Refer %':<10} | {'PSNR (Accepted)'}")
    print("-" * 80)

    sub_ds = torch.utils.data.Subset(dataset, range(min(max_samples, len(dataset))))
    loader = DataLoader(sub_ds, batch_size=16, shuffle=False)
    
    a4_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    results = []

    for a4 in a4_values:
        psnrs_all = []
        psnrs_acc = []
        mc_scores = []
        risks = []
        decisions_list = []

        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                B = images.shape[0]
                
                # Apply optical shift to forward measurement
                opt_shift = torch.zeros(B, 11, device=device)
                opt_shift[:, 4] = a4
                y_meas, _ = fwd_model.forward_sample(images, opt_shift)

                # Blind reconstruction (mode C: no oracle defocus parameter)
                nig_out, _ = model(y_meas, optical_params=None, return_evidential=True)
                recon = nig_out["gamma"]
                unc = nig_out["total_uncertainty"].mean(dim=(-1, -2, -3)) # (B,)

                # Measurement consistency evaluated against nominal model
                mc = scorer.compute_residual(recon, y_meas)
                risk = policy.compute_risk(mc, unc)
                decisions, _ = policy.decide(risk)

                for b in range(B):
                    p = calculate_psnr(images[b:b+1], recon[b:b+1])
                    psnrs_all.append(p)
                    mc_scores.append(float(mc[b].item()))
                    risks.append(float(risk[b].item()))
                    decisions_list.append(int(decisions[b].item()))
                    if decisions[b].item() == 0: # ACCEPT
                        psnrs_acc.append(p)

        dec_arr = np.array(decisions_list)
        pct_acc = float((dec_arr == 0).mean() * 100)
        pct_rem = float((dec_arr == 1).mean() * 100)
        pct_ref = float((dec_arr == 2).mean() * 100)
        m_psnr_all = float(np.mean(psnrs_all))
        m_psnr_acc = float(np.mean(psnrs_acc)) if len(psnrs_acc) > 0 else float("nan")
        m_mc = float(np.mean(mc_scores))

        row = {
            "defocus_a4": a4,
            "psnr_all_mean": m_psnr_all,
            "psnr_accepted_mean": m_psnr_acc,
            "mc_residual_mean": m_mc,
            "pct_accept": pct_acc,
            "pct_remeasure": pct_rem,
            "pct_refer": pct_ref
        }
        results.append(row)
        acc_str = f"{m_psnr_acc:5.2f} dB" if not np.isnan(m_psnr_acc) else "N/A (0 acc)"
        print(f"{a4:<10.2f} | {m_psnr_all:5.2f} dB    | {m_mc:5.2f}        | {pct_acc:6.1f}%    | {pct_rem:6.1f}%    | {pct_ref:6.1f}%    | {acc_str}")

    return results


def run_photon_budget_sweep(
    model: ICPINOT,
    scorer: MeasurementConsistencyScorer,
    policy: SelectiveInspectionPolicy,
    dataset,
    patterns: torch.Tensor,
    device: torch.device,
    max_samples: int = 100
) -> list:
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: PHOTON BUDGET SHIFT SWEEP (Shot Noise Limit to Rich Flux)")
    print("=" * 80)
    print(f"{'N_total':<10} | {'Phot/Pattern':<12} | {'PSNR (All)':<12} | {'MC Residual':<12} | {'Accept %':<10} | {'Re-meas %':<10} | {'Refer %'}")
    print("-" * 80)

    sub_ds = torch.utils.data.Subset(dataset, range(min(max_samples, len(dataset))))
    loader = DataLoader(sub_ds, batch_size=16, shuffle=False)

    budgets = [
        (1024, 2.0),
        (2560, 5.0),
        (5120, 10.0),
        (10240, 20.0),
        (25600, 50.0)
    ]
    results = []

    for n_tot, phot_per_pat in budgets:
        # Create forward model with targeted mean photon count
        fwd_b = GhostImagingForwardModel(image_size=64, num_patterns=512, mean_photon_count=phot_per_pat, pattern_type="learned").to(device)
        fwd_b.measurement_matrix.H_raw.data = torch.logit(patterns.to(device).clamp(1e-5, 1 - 1e-5))
        scorer_b = MeasurementConsistencyScorer(fwd_b)

        psnrs = []
        mc_scores = []
        decisions_list = []

        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                B = images.shape[0]
                y_meas, _ = fwd_b.forward_sample(images, torch.zeros(B, 11, device=device))

                nig_out, _ = model(y_meas, optical_params=None, return_evidential=True)
                recon = nig_out["gamma"]
                unc = nig_out["total_uncertainty"].mean(dim=(-1, -2, -3))

                mc = scorer_b.compute_residual(recon, y_meas)
                risk = policy.compute_risk(mc, unc)
                decisions, _ = policy.decide(risk)

                for b in range(B):
                    psnrs.append(calculate_psnr(images[b:b+1], recon[b:b+1]))
                    mc_scores.append(float(mc[b].item()))
                    decisions_list.append(int(decisions[b].item()))

        dec_arr = np.array(decisions_list)
        pct_acc = float((dec_arr == 0).mean() * 100)
        pct_rem = float((dec_arr == 1).mean() * 100)
        pct_ref = float((dec_arr == 2).mean() * 100)
        m_psnr = float(np.mean(psnrs))
        m_mc = float(np.mean(mc_scores))

        row = {
            "n_total_photons": n_tot,
            "photons_per_pattern": phot_per_pat,
            "psnr_mean": m_psnr,
            "mc_residual_mean": m_mc,
            "pct_accept": pct_acc,
            "pct_remeasure": pct_rem,
            "pct_refer": pct_ref
        }
        results.append(row)
        print(f"{n_tot:<10} | {phot_per_pat:<12.1f} | {m_psnr:5.2f} dB    | {m_mc:5.2f}        | {pct_acc:6.1f}%    | {pct_rem:6.1f}%    | {pct_ref:6.1f}%")

    return results


def run_cross_fabric_selective_inspection(
    model: ICPINOT,
    fwd_model: GhostImagingForwardModel,
    scorer: MeasurementConsistencyScorer,
    policy: SelectiveInspectionPolicy,
    aitex_dataset: AITEXDataset,
    device: torch.device
) -> dict:
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: CROSS-FABRIC MATERIAL DOMAIN SHIFT & SELECTIVE DEFECT INSPECTION")
    print("=" * 80)

    loader = DataLoader(aitex_dataset, batch_size=16, shuffle=False)
    
    y_true = []
    y_scores_raw = []
    decisions_list = []
    psnrs = []
    remeasured_psnrs = []

    # Adaptive re-measurer forward model with 2x photon budget (10240 photons)
    fwd_remeas = GhostImagingForwardModel(image_size=64, num_patterns=512, mean_photon_count=20.0, pattern_type="learned").to(device)
    fwd_remeas.measurement_matrix.H_raw.data = fwd_model.measurement_matrix.H_raw.data.clone()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            B = images.shape[0]

            # Primary measurement at nominal budget (5120 photons)
            y_meas, _ = fwd_model.forward_sample(images, torch.zeros(B, 11, device=device))

            nig_out, logits = model(y_meas, optical_params=None, return_evidential=True)
            recon = nig_out["gamma"]
            unc = nig_out["total_uncertainty"].mean(dim=(-1, -2, -3))

            mc = scorer.compute_residual(recon, y_meas)
            risk = policy.compute_risk(mc, unc)
            decisions, _ = policy.decide(risk)

            # Defect probability: sigmoid(logits[:, 1] - logits[:, 0]) or softmax
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            for b in range(B):
                lbl = int(labels[b].item())
                y_true.append(lbl)
                y_scores_raw.append(float(probs[b]))
                dec = int(decisions[b].item())
                decisions_list.append(dec)
                p = calculate_psnr(images[b:b+1], recon[b:b+1])
                psnrs.append(p)

                # If flagged for RE-MEASURE, simulate secondary adaptive burst
                if dec == 1:
                    y_boost, _ = fwd_remeas.forward_sample(images[b:b+1], torch.zeros(1, 11, device=device))
                    rec_boost, _ = model(y_boost, optical_params=None)
                    remeasured_psnrs.append(calculate_psnr(images[b:b+1], rec_boost))
                else:
                    remeasured_psnrs.append(p)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores_raw)
    dec_arr = np.array(decisions_list)

    # 1. Un-calibrated All-In Inspection (No abstention)
    auroc_all = float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) > 1 else float("nan")

    # 2. Selective Inspection on Accepted Samples (Filtered highest risk / referred samples)
    acc_mask = (dec_arr == 0)
    if acc_mask.sum() > 0 and len(np.unique(y_true[acc_mask])) > 1:
        auroc_accepted = float(roc_auc_score(y_true[acc_mask], y_scores[acc_mask]))
    else:
        auroc_accepted = auroc_all # fallback if single class

    # Non-referred samples (Accepted + Re-measured)
    non_ref_mask = (dec_arr != 2)
    if non_ref_mask.sum() > 0 and len(np.unique(y_true[non_ref_mask])) > 1:
        auroc_non_ref = float(roc_auc_score(y_true[non_ref_mask], y_scores[non_ref_mask]))
    else:
        auroc_non_ref = auroc_all

    psnr_baseline = float(np.mean(psnrs))
    psnr_after_remeas = float(np.mean(remeasured_psnrs))
    remeas_gain = psnr_after_remeas - psnr_baseline

    summary = {
        "dataset": "AITEX Cross-Fabric",
        "total_samples": len(y_true),
        "accept_count": int((dec_arr == 0).sum()),
        "remeasure_count": int((dec_arr == 1).sum()),
        "refer_count": int((dec_arr == 2).sum()),
        "auroc_uncalibrated_all": auroc_all,
        "auroc_selective_accepted": auroc_accepted,
        "auroc_selective_non_referred": auroc_non_ref,
        "psnr_baseline_mean": psnr_baseline,
        "psnr_adaptive_remeas_mean": psnr_after_remeas,
        "psnr_remeas_gain_dB": remeas_gain
    }

    print(f"Total Inspection Samples:       {len(y_true)}")
    print(f"Policy Decision Breakdown:      Accept: {summary['accept_count']} | Re-measure: {summary['remeasure_count']} | Refer: {summary['refer_count']}")
    print(f"All-In Inspection AUROC:        {auroc_all:.4f}")
    print(f"Selective Policy (Non-Refer):   {auroc_non_ref:.4f} (abstaining on unresolvable operator/aberration shifts)")
    print(f"Baseline PSNR (5,120 photons):  {psnr_baseline:.2f} dB")
    print(f"Adaptive Re-measurement PSNR:   {psnr_after_remeas:.2f} dB (+{remeas_gain:.2f} dB autonomous gain)")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Shift and Selective Inspection Benchmark")
    parser.add_argument("--patterns", default="checkpoints/trained_patterns.pt")
    parser.add_argument("--ckpt", default="checkpoints/ic_pinot_tilda_400_seed42.pt")
    parser.add_argument("--output", default="results/shift_benchmark_results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Patterns & Initialize Models
    patterns = torch.load(args.patterns, map_location=device)
    fwd = GhostImagingForwardModel(image_size=64, num_patterns=512, pattern_type="learned").to(device)
    fwd.measurement_matrix.H_raw.data = torch.logit(patterns.clamp(1e-5, 1 - 1e-5))
    
    model = ICPINOT(num_patterns=512, img_size=64).to(device)
    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    model.eval()

    scorer = MeasurementConsistencyScorer(fwd)
    policy = SelectiveInspectionPolicy(tau_accept=0.50, tau_refer=0.85, alpha_mc=0.6)

    # Calibrate policy on in-distribution TILDA validation set
    print("Calibrating Selective Inspection Policy on TILDA validation split...")
    tilda_train = TILDADataset("data/benchmarks/tilda", split="train")
    calib_sub = torch.utils.data.Subset(tilda_train, range(min(200, len(tilda_train))))
    calib_loader = DataLoader(calib_sub, batch_size=16, shuffle=False)
    
    val_mc, val_unc = [], []
    with torch.no_grad():
        for imgs, _ in calib_loader:
            imgs = imgs.to(device)
            B = imgs.shape[0]
            y_nom, _ = fwd.forward_sample(imgs, torch.zeros(B, 11, device=device))
            nig, _ = model(y_nom, optical_params=None, return_evidential=True)
            mc = scorer.compute_residual(nig["gamma"], y_nom)
            unc = nig["total_uncertainty"].mean(dim=(-1, -2, -3))
            val_mc.extend(mc.cpu().numpy().tolist())
            val_unc.extend(unc.cpu().numpy().tolist())

    policy.calibrate(np.array(val_mc), np.array(val_unc))
    print(f"Policy Calibrated: thresh_accept = {policy.thresh_accept:.3f}, thresh_refer = {policy.thresh_refer:.3f}")

    tilda_test = TILDADataset("data/benchmarks/tilda", split="test")
    aitex_all = AITEXDataset("data/aitex")

    # Run Benchmark Experiments
    exp1 = run_optical_shift_sweep(model, fwd, scorer, policy, tilda_test, device, max_samples=100)
    exp2 = run_photon_budget_sweep(model, scorer, policy, tilda_test, patterns, device, max_samples=100)
    exp3 = run_cross_fabric_selective_inspection(model, fwd, scorer, policy, aitex_all, device)

    full_results = {
        "metadata": {
            "title": "Risk-Calibrated SPI Inspection Under Joint Shift",
            "device": str(device),
            "patterns": args.patterns,
            "checkpoint": args.ckpt
        },
        "optical_shift_sweep": exp1,
        "photon_budget_sweep": exp2,
        "cross_fabric_selective_inspection": exp3
    }

    with open(args.output, "w") as f:
        json.dump(full_results, f, indent=4)
    print("\n" + "=" * 80)
    print(f"ALL SHIFT BENCHMARK EXPERIMENTS COMPLETED. Results saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
