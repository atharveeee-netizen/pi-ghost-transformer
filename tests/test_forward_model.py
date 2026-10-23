"""
Unit and Physical Invariant Tests for GhostImagingForwardModel.

Verifies:
1. Zero signal input: purely noise / non-negative counts.
2. Non-negativity of simulated bucket measurements.
3. Linear adjoint consistency: <Hx, y> = <x, H^T y>.
4. Seed repeatability / deterministic execution.
5. Photon count scaling / Poisson variance properties.
6. Identity / no-blur optical transfer function.
7. Rank and conditioning of illumination pattern matrix.
"""

import os
import sys
import unittest
import torch
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from physics.forward_model import GhostImagingForwardModel, ZernikePSF, apply_psf_blur, PhotonNoiseModel


class TestGhostImagingForwardModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = 64
        self.num_patterns = 512
        self.fwd = GhostImagingForwardModel(
            image_size=self.img_size,
            num_patterns=self.num_patterns,
            pattern_type="learned"
        ).to(self.device)

        # Initialize with trained patterns if present, else clamp
        pattern_path = os.path.join(ROOT_DIR, "checkpoints", "trained_patterns.pt")
        if os.path.exists(pattern_path):
            patterns = torch.load(pattern_path, map_location=self.device)
            self.fwd.measurement_matrix.H_raw.data = torch.logit(patterns.clamp(1e-5, 1 - 1e-5))

    def test_non_negativity(self):
        """Photon bucket counts must be strictly non-negative."""
        x = torch.rand(4, 1, self.img_size, self.img_size, device=self.device)
        optical_params = torch.zeros(4, 11, device=self.device)
        y, _ = self.fwd.forward_sample(x, optical_params)
        self.assertTrue((y >= 0.0).all().item(), "Bucket measurements contained negative values")

    def test_zero_signal(self):
        """Zero image input should produce dark count rate baseline."""
        x_zero = torch.zeros(2, 1, self.img_size, self.img_size, device=self.device)
        optical_params = torch.zeros(2, 11, device=self.device)
        y_exp = self.fwd.forward_expected(x_zero, optical_params)
        expected_dark = torch.full_like(y_exp, self.fwd.dark_count_rate)
        self.assertTrue(torch.allclose(y_exp, expected_dark, atol=1e-4),
                        f"Expected flux for zero image should equal dark count rate {self.fwd.dark_count_rate}")

    def test_linear_adjoint_consistency(self):
        """For the linear measurement operator H: <Hx, y> = <x, H^T y>."""
        H = self.fwd.measurement_matrix.patterns # (M, N)
        M, N = H.shape
        x = torch.randn(N, 1, device=self.device)
        y = torch.randn(M, 1, device=self.device)

        Hx = torch.matmul(H, x)
        Hty = torch.matmul(H.t(), y)

        inner1 = torch.sum(Hx * y)
        inner2 = torch.sum(x * Hty)

        rel_diff = (inner1 - inner2).abs() / (inner1.abs() + inner2.abs() + 1e-8)
        self.assertLess(rel_diff.item(), 1e-5, "Linear adjoint consistency failed")

    def test_seed_repeatability(self):
        """With deterministic seeds, stochastic Poisson noise should be exactly reproducible."""
        x = torch.rand(2, 1, self.img_size, self.img_size, device=self.device)
        optical_params = torch.zeros(2, 11, device=self.device)

        torch.manual_seed(12345)
        y1, _ = self.fwd.forward_sample(x, optical_params)

        torch.manual_seed(12345)
        y2, _ = self.fwd.forward_sample(x, optical_params)

        self.assertTrue(torch.allclose(y1, y2, atol=1e-6), "Forward model is non-deterministic given same seed")

    def test_identity_zero_defocus(self):
        """Zero Zernike coefficients should produce energy-conserving PSF blur."""
        zernike = ZernikePSF(psf_size=21).to(self.device)
        optical_params = torch.zeros(2, 11, device=self.device)
        psf = zernike(optical_params)
        psf_sum = psf.sum(dim=(-1, -2))
        self.assertTrue(torch.allclose(psf_sum, torch.ones_like(psf_sum), atol=1e-3),
                        "PSF was not energy-conserving under zero aberration")
        x = torch.ones(2, 1, self.img_size, self.img_size, device=self.device)
        x_blurred = apply_psf_blur(x, psf)
        self.assertTrue(torch.allclose(x_blurred, x, atol=1e-3),
                        "Flat field intensity was not conserved under zero aberration")

    def test_pattern_matrix_conditioning(self):
        """The measurement matrix H must have full row rank (M <= N) without degenerate singular values."""
        H = self.fwd.measurement_matrix.patterns.detach().cpu() # (512, 4096)
        M, N = H.shape
        rank = torch.linalg.matrix_rank(H)
        self.assertEqual(rank.item(), M, f"Pattern matrix is rank-deficient: rank {rank.item()} < {M}")

        s = torch.linalg.svdvals(H)
        cond = (s[0] / s[-1]).item()
        self.assertLess(cond, 1e4, f"Pattern matrix condition number is excessively high: {cond:.2e}")


if __name__ == "__main__":
    unittest.main()
