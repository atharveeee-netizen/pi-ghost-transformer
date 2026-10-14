"""
Master Orchestration Engine for IC-PINOT Publication Campaign.
Sequences all 5 pillars:
  Pillar 1: Flagship IC-PINOT Multi-Seed Training across 17 Benchmarks
  Pillar 2: Direct Head-to-Head Baseline Suite (DGI, Vanilla Transformer, IC-PINOT)
  Pillar 3: 3D Physical Stress Matrix (Photon flux, Defocus, Sub-Nyquist sampling)
  Pillar 4: Evidential Uncertainty Calibration & Orthogonal Dissociation
  Pillar 5: Zero-Gradient In-Context Adaptation Benchmark
"""

import os
import sys
import argparse
import datetime
import json

from train_ic_pinot_master import run_full_suite
from benchmark_baselines import main as run_baselines
from benchmark_stress_matrix import evaluate_stress_matrix
from benchmark_uncertainty_calibration import run_uncertainty_calibration
from benchmark_in_context_adaptation import evaluate_in_context_adaptation


def main():
    parser = argparse.ArgumentParser(description="Master IC-PINOT Orchestrator")
    parser.add_argument("--pillar", type=int, choices=[1, 2, 3, 4, 5], default=None, help="Run specific pillar")
    parser.add_argument("--epochs", type=int, default=150, help="Epochs for Pillar 1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds for Pillar 1")
    parser.add_argument("--categories", type=str, nargs="+", default=None, help="MVTec categories subset")
    parser.add_argument("--all", action="store_true", help="Run entire 5-pillar suite sequentially")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    start_time = datetime.datetime.now()
    print("=" * 80)
    print(f"IC-PINOT MASTER PUBLICATION CAMPAIGN INITIALIZED at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    if args.pillar == 1 or args.all:
        print("\n>>> LAUNCHING PILLAR 1: FLAGSHIP IC-PINOT TRAINING")
        run_full_suite(epochs=args.epochs, seeds=args.seeds, categories=args.categories)

    if args.pillar == 2 or args.all:
        print("\n>>> LAUNCHING PILLAR 2: HEAD-TO-HEAD BASELINE BENCHMARKS")
        run_baselines()

    if args.pillar == 3 or args.all:
        print("\n>>> LAUNCHING PILLAR 3: 3D PHYSICAL STRESS MATRIX")
        evaluate_stress_matrix()

    if args.pillar == 4 or args.all:
        print("\n>>> LAUNCHING PILLAR 4: UNCERTAINTY CALIBRATION & DISSOCIATION")
        run_uncertainty_calibration()

    if args.pillar == 5 or args.all:
        print("\n>>> LAUNCHING PILLAR 5: ZERO-GRADIENT IN-CONTEXT ADAPTATION")
        evaluate_in_context_adaptation()

    elapsed = datetime.datetime.now() - start_time
    print("\n" + "=" * 80)
    print(f"IC-PINOT CAMPAIGN SUCCESSFULLY COMPLETED! Total Duration: {elapsed}")
    print("All artifacts and benchmark JSONs saved in results/")
    print("=" * 80)


if __name__ == "__main__":
    main()
