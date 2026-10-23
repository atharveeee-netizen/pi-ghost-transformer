"""
Comprehensive Defect Inspection Benchmark under Selective Operational Policy.
Measures AUROC, AUPRC, Recall, Precision, False Negative Rate (FNR),
and Risk-Coverage Curves compared with Matched-Coverage Random Rejection.
"""

import os
import sys
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from torch.utils.data import DataLoader, Subset

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from physics.forward_model import GhostImagingForwardModel
from models.ic_pinot import ICPINOT
from models.selective_inspection import MeasurementConsistencyScorer, SelectiveInspectionPolicy
from data.benchmark_loader import TILDADataset, DAGMDataset
from data.aitex_loader import AITEXDataset


def evaluate_defect_selective(dataset_name, dataset, model, fwd, scorer, policy, device, max_samples=300):
    print(f"\n{'='*80}")
    print(f"DEFECT INSPECTION & SELECTIVE RISK BENCHMARK: {dataset_name}")
    print(f"{'='*80}")

    sub_ds = Subset(dataset, range(min(max_samples, len(dataset))))
    loader = DataLoader(sub_ds, batch_size=20, shuffle=False)

    y_true = []
    y_probs = []
    mc_scores = []
    unc_scores = []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            B = imgs.shape[0]

            y_meas, _ = fwd.forward_sample(imgs, torch.zeros(B, 11, device=device))
            nig_out, logits = model(y_meas, optical_params=None, return_evidential=True)
            rec = nig_out["gamma"]
            unc = nig_out["total_uncertainty"].mean(dim=(-1, -2, -3))

            mc = scorer.compute_residual(rec, y_meas)
            probs = torch.softmax(logits, dim=1)[:, 1]

            y_true.extend(lbls.numpy().tolist())
            y_probs.extend(probs.cpu().numpy().tolist())
            mc_scores.extend(mc.cpu().numpy().tolist())
            unc_scores.extend(unc.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    mc_scores = np.array(mc_scores)
    unc_scores = np.array(unc_scores)

    # Defect prevalence
    n_total = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n_total - n_pos
    prevalence = n_pos / n_total

    print(f"Total Samples Evaluated: {n_total} (Defect-Free: {n_neg}, Defective: {n_pos}, Prevalence: {prevalence*100:.1f}%)")

    # All-in metrics (Coverage = 100%)
    auroc_all = float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else float("nan")
    auprc_all = float(average_precision_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else float("nan")
    
    y_pred_all = (y_probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_all, labels=[0, 1]).ravel()
    recall_all = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision_all = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    fnr_all = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0

    print(f"All-In Inspection (100% Coverage):")
    print(f"  AUROC: {auroc_all:.4f} | AUPRC: {auprc_all:.4f} | Recall: {recall_all:.4f} | Precision: {precision_all:.4f} | FNR: {fnr_all:.4f}")

    # Standardized composite risk score
    z_mc = (mc_scores - np.mean(mc_scores)) / (np.std(mc_scores) + 1e-8)
    z_unc = (unc_scores - np.mean(unc_scores)) / (np.std(unc_scores) + 1e-8)
    risk_scores = 0.6 * z_mc + 0.4 * z_unc

    # Sweep Coverage levels
    coverage_levels = [1.0, 0.90, 0.80, 0.70, 0.60, 0.50]
    selective_results = {}

    print(f"\n{'Coverage':<10} | {'Policy AUROC':<14} | {'Policy FNR':<12} | {'Random AUROC':<14} | {'Random FNR'}")
    print("-" * 70)

    for cov in coverage_levels:
        n_keep = int(n_total * cov)
        
        # Proposed Selective Policy: Keep lowest risk samples
        keep_order = np.argsort(risk_scores)
        idx_policy = keep_order[:n_keep]

        # Matched Coverage Random Baseline (average over 5 random seeds)
        rand_aurocs, rand_fnrs = [], []
        for s in range(5):
            np.random.seed(42 + s)
            idx_rand = np.random.choice(n_total, size=n_keep, replace=False)
            if len(np.unique(y_true[idx_rand])) > 1:
                rand_aurocs.append(roc_auc_score(y_true[idx_rand], y_probs[idx_rand]))
            y_p_r = (y_probs[idx_rand] >= 0.5).astype(int)
            tn_r, fp_r, fn_r, tp_r = confusion_matrix(y_true[idx_rand], y_p_r, labels=[0, 1]).ravel()
            fnr_r = float(fn_r / (tp_r + fn_r)) if (tp_r + fn_r) > 0 else 0.0
            rand_fnrs.append(fnr_r)

        rand_auroc_mean = float(np.mean(rand_aurocs)) if len(rand_aurocs) > 0 else float("nan")
        rand_fnr_mean = float(np.mean(rand_fnrs))

        # Evaluate policy accepted subset
        if len(np.unique(y_true[idx_policy])) > 1:
            pol_auroc = float(roc_auc_score(y_true[idx_policy], y_probs[idx_policy]))
        else:
            pol_auroc = float("nan")

        y_p_pol = (y_probs[idx_policy] >= 0.5).astype(int)
        tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_true[idx_policy], y_p_pol, labels=[0, 1]).ravel()
        pol_fnr = float(fn_p / (tp_p + fn_p)) if (tp_p + fn_p) > 0 else 0.0
        pol_recall = float(tp_p / (tp_p + fn_p)) if (tp_p + fn_p) > 0 else 0.0

        selective_results[f"cov_{int(cov*100)}"] = {
            "coverage": cov,
            "policy_auroc": pol_auroc,
            "policy_fnr": pol_fnr,
            "policy_recall": pol_recall,
            "random_auroc": rand_auroc_mean,
            "random_fnr": rand_fnr_mean
        }

        print(f"{cov*100:5.0f}%     | {pol_auroc:10.4f}     | {pol_fnr:10.4f}   | {rand_auroc_mean:10.4f}     | {rand_fnr_mean:10.4f}")

    return {
        "dataset": dataset_name,
        "total_samples": n_total,
        "prevalence": prevalence,
        "all_in": {
            "auroc": auroc_all,
            "auprc": auprc_all,
            "recall": recall_all,
            "precision": precision_all,
            "fnr": fnr_all
        },
        "selective_coverage_sweep": selective_results
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patterns = torch.load(os.path.join(ROOT_DIR, "checkpoints", "trained_patterns.pt"), map_location=device)
    fwd = GhostImagingForwardModel(image_size=64, num_patterns=512, pattern_type="learned").to(device)
    fwd.measurement_matrix.H_raw.data = torch.logit(patterns.clamp(1e-5, 1 - 1e-5))

    scorer = MeasurementConsistencyScorer(fwd)
    policy = SelectiveInspectionPolicy(tau_accept=0.5, tau_refer=0.85, alpha_mc=0.6)

    results = {}

    # 1. TILDA 400
    model_tilda = ICPINOT(num_patterns=512, img_size=64).to(device)
    model_tilda.load_state_dict(torch.load(os.path.join(ROOT_DIR, "checkpoints", "ic_pinot_tilda_400_seed42.pt"), map_location=device), strict=False)
    model_tilda.eval()
    ds_tilda = TILDADataset(os.path.join(ROOT_DIR, "data", "benchmarks", "tilda"), split="test")
    results["TILDA 400"] = evaluate_defect_selective("TILDA 400", ds_tilda, model_tilda, fwd, scorer, policy, device, max_samples=300)

    # 2. DAGM 2007
    model_dagm = ICPINOT(num_patterns=512, img_size=64).to(device)
    model_dagm.load_state_dict(torch.load(os.path.join(ROOT_DIR, "checkpoints", "ic_pinot_dagm_2007_seed42.pt"), map_location=device), strict=False)
    model_dagm.eval()
    ds_dagm = DAGMDataset(os.path.join(ROOT_DIR, "data", "benchmarks", "dagm"), split="test")
    results["DAGM 2007"] = evaluate_defect_selective("DAGM 2007", ds_dagm, model_dagm, fwd, scorer, policy, device, max_samples=300)

    # 3. AITEX (Cross-Fabric Domain Shift)
    ds_aitex = AITEXDataset(os.path.join(ROOT_DIR, "data", "aitex"), split="all")
    results["AITEX Cross-Fabric"] = evaluate_defect_selective("AITEX Cross-Fabric (Zero-Shot)", ds_aitex, model_tilda, fwd, scorer, policy, device, max_samples=247)

    out_file = os.path.join(os.path.dirname(__file__), "results", "defect_inspection_selective_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nAll defect inspection results successfully saved to: {out_file}\n")


if __name__ == "__main__":
    main()
