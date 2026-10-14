"""
Hardware-Software Co-Design Analysis Tool.
Analyzes the optical measurement matrix H:
  1. Learned Differentiable Patterns vs. Deterministic Hadamard vs. Random
  2. 2D Spatial Frequency Power Spectral Density (PSD)
  3. Condition Number and Singular Value Spectrum
"""

import torch
import numpy as np
import os
import sys
import json

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def analyze_measurement_matrix(patterns: torch.Tensor, name: str = "Learned", img_size: int = 64) -> dict:
    """
    Args:
        patterns: (M, N^2) tensor of illumination patterns
        name: Name of the pattern family
        img_size: Spatial resolution
    """
    M, N_sq = patterns.shape
    H_np = patterns.detach().cpu().numpy()
    
    # 1. Singular value decomposition for sensing matrix conditioning
    try:
        U, S, Vt = np.linalg.svd(H_np, full_matrices=False)
        cond_num = float(S[0] / (S[-1] + 1e-12))
        effective_rank = int(np.sum(S > 1e-3))
    except Exception:
        cond_num = float("nan")
        effective_rank = M

    # 2. 2D Power Spectral Density (Fourier Energy Concentration)
    patterns_2d = H_np.reshape(M, img_size, img_size)
    fft_patterns = np.fft.fftshift(np.fft.fft2(patterns_2d), axes=(-2, -1))
    psd = np.mean(np.abs(fft_patterns) ** 2, axis=0)
    
    # Low vs High frequency ratio
    cy, cx = img_size // 2, img_size // 2
    r_cutoff = img_size // 4
    y, x = np.ogrid[:img_size, :img_size]
    mask_low = ((x - cx)**2 + (y - cy)**2) <= r_cutoff**2
    
    low_freq_energy = float(np.sum(psd[mask_low]))
    total_energy = float(np.sum(psd))
    high_freq_ratio = float((total_energy - low_freq_energy) / (total_energy + 1e-12))
    
    metrics = {
        "name": name,
        "num_patterns": M,
        "compression_ratio": float(M / N_sq),
        "condition_number": cond_num,
        "effective_rank": effective_rank,
        "high_freq_energy_ratio": high_freq_ratio,
        "mean_intensity": float(np.mean(H_np)),
        "std_intensity": float(np.std(H_np))
    }
    
    print(f"[{name}] Rank: {effective_rank}/{M} | High-Freq Energy: {high_freq_ratio:.1%} | Cond Num: {cond_num:.2e}")
    return metrics


def main():
    os.makedirs("results", exist_ok=True)
    from physics.forward_model import MeasurementMatrix
    
    print("=" * 65)
    print("OPTICAL HARDWARE-SOFTWARE CO-DESIGN BENCHMARK")
    print("=" * 65)
    
    results = {}
    
    # 1. Learned Differentiable Matrix
    mat_learned = MeasurementMatrix(num_patterns=512, image_size=64, pattern_type="learned")
    results["Learned_Differentiable"] = analyze_measurement_matrix(mat_learned.patterns, "Learned Co-Designed")
    
    # 2. Deterministic Hadamard Matrix
    mat_hadamard = MeasurementMatrix(num_patterns=512, image_size=64, pattern_type="hadamard")
    results["Deterministic_Hadamard"] = analyze_measurement_matrix(mat_hadamard.patterns, "Deterministic Hadamard")
    
    # 3. Random Binary Matrix
    mat_random = MeasurementMatrix(num_patterns=512, image_size=64, pattern_type="random")
    results["Random_Binary"] = analyze_measurement_matrix(mat_random.patterns, "Random Binary")
    
    out_file = "results/hardware_codesign_analysis.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nCo-design analysis saved to {out_file}")


if __name__ == "__main__":
    main()
