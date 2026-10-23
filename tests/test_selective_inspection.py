"""
Unit tests for Risk-Calibrated Selective Inspection Policy.
"""

import os
import sys
import unittest
import torch
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from physics.forward_model import GhostImagingForwardModel
from models.selective_inspection import MeasurementConsistencyScorer, SelectiveInspectionPolicy


class TestSelectiveInspection(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fwd = GhostImagingForwardModel(image_size=64, num_patterns=512, pattern_type="learned").to(self.device)
        pattern_path = os.path.join(ROOT_DIR, "checkpoints", "trained_patterns.pt")
        if os.path.exists(pattern_path):
            patterns = torch.load(pattern_path, map_location=self.device)
            self.fwd.measurement_matrix.H_raw.data = torch.logit(patterns.clamp(1e-5, 1 - 1e-5))

    def test_measurement_consistency_under_clean_and_defocused(self):
        """Measurement consistency residual must be significantly higher under optical defocus shift."""
        scorer = MeasurementConsistencyScorer(self.fwd)
        x = torch.rand(4, 1, 64, 64, device=self.device)
        
        # Nominal zero-defocus measurements
        opt_nominal = torch.zeros(4, 11, device=self.device)
        y_nominal, _ = self.fwd.forward_sample(x, opt_nominal)

        # Defocused measurements (defocus coefficient a4 = 2.5)
        opt_shifted = torch.zeros(4, 11, device=self.device)
        opt_shifted[:, 4] = 2.5
        y_shifted, _ = self.fwd.forward_sample(x, opt_shifted)

        # Residuals evaluated blindly against nominal model using same x
        r_nominal = scorer.compute_residual(x, y_nominal, opt_nominal)
        r_shifted = scorer.compute_residual(x, y_shifted, opt_nominal)

        self.assertGreater(r_shifted.mean().item(), r_nominal.mean().item(),
                           "Measurement consistency did not increase under optical defocus shift")

    def test_policy_calibration_and_decision(self):
        """Calibration computes empirical quantiles and allocates three decision tiers."""
        policy = SelectiveInspectionPolicy(tau_accept=0.50, tau_refer=0.85, alpha_mc=0.6)
        val_mc = np.random.normal(loc=1.0, scale=0.2, size=200)
        val_unc = np.random.normal(loc=0.05, scale=0.01, size=200)

        policy.calibrate(val_mc, val_unc)
        self.assertTrue(policy.is_calibrated)
        self.assertLess(policy.thresh_accept, policy.thresh_refer)

        # Test evaluation with known distribution
        test_mc = torch.from_numpy(np.random.normal(loc=1.0, scale=0.2, size=100)).float().to(self.device)
        test_unc = torch.from_numpy(np.random.normal(loc=0.05, scale=0.01, size=100)).float().to(self.device)

        risk = policy.compute_risk(test_mc, test_unc)
        decisions, counts = policy.decide(risk)

        self.assertEqual(counts["total"], 100)
        self.assertGreater(counts["accept"], 0)
        self.assertGreater(counts["re_measure"], 0)
        self.assertGreater(counts["refer"], 0)
        self.assertEqual(counts["accept"] + counts["re_measure"] + counts["refer"], 100)


if __name__ == "__main__":
    unittest.main()
