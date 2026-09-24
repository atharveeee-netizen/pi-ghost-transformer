"""
Complete Differentiable Forward Model for Ghost Imaging.

Implements the full image-formation pipeline:

    x_clean ──► [PSF Blur] ──► [Pattern Modulation] ──► [Bucket Detection] ──► [Poisson Noise] ──► y

Forward model equation:
    y_m = Poisson( N̄ · Σᵢ H_m(i) · (PSF ⊛ x)(i) + d ) + N(0, σ²)

where:
    x       = clean fabric image
    PSF     = point spread function (Zernike-based)
    H_m     = m-th illumination pattern
    y_m     = m-th bucket detector measurement
    N̄       = mean photon count
    d       = dark count rate

The differentiable version (for training) replaces Poisson sampling
with its expected value and uses Poisson NLL in the loss function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

from .psf import ZernikePSF, apply_psf_blur
from .noise import PhotonNoiseModel, ScatteringModel


class MeasurementMatrix(nn.Module):
    """
    Illumination pattern matrix for single-pixel / ghost imaging.

    Supports three modes:
        - "hadamard": Fixed Hadamard patterns (deterministic)
        - "random": Fixed random binary patterns
        - "learned": Learnable patterns (co-optimized with reconstruction)

    The measurement matrix H has shape (M, N²), where:
        M = number of measurements (patterns)
        N² = number of image pixels (flattened)

    Physical constraint: H_ij ∈ [0, 1] (light intensity is non-negative).

    Args:
        num_patterns: Number of illumination patterns M.
        image_size: Image resolution N (produces N² pixel flattening).
        pattern_type: One of "hadamard", "random", "learned".
    """

    def __init__(
        self,
        num_patterns: int = 512,
        image_size: int = 64,
        pattern_type: str = "learned",
    ):
        super().__init__()
        self.num_patterns = num_patterns
        self.image_size = image_size
        self.num_pixels = image_size * image_size
        self.pattern_type = pattern_type

        if pattern_type == "hadamard":
            H = self._hadamard_patterns()
            self.register_buffer("H", H)
        elif pattern_type == "random":
            H = torch.rand(num_patterns, self.num_pixels)
            H = (H > 0.5).float()
            self.register_buffer("H", H)
        elif pattern_type == "learned":
            # Learnable parameters (unconstrained); sigmoid applied in forward
            self.H_raw = nn.Parameter(torch.randn(num_patterns, self.num_pixels) * 0.1)
        else:
            raise ValueError(f"Unknown pattern_type: {pattern_type}")

    def _hadamard_patterns(self) -> torch.Tensor:
        """Generate Hadamard basis patterns (requires N² to be a power of 2)."""
        from scipy.linalg import hadamard

        n = self.num_pixels
        # Find next power of 2
        n_pow2 = 2 ** int(np.ceil(np.log2(n)))
        H_full = torch.from_numpy(hadamard(n_pow2).astype(np.float32))
        # Select subset and normalize to [0, 1]
        H = H_full[:self.num_patterns, :n]
        H = (H + 1.0) / 2.0  # Map {-1, 1} → {0, 1}
        return H

    @property
    def patterns(self) -> torch.Tensor:
        """Get the measurement patterns with physical constraints applied."""
        if self.pattern_type == "learned":
            return torch.sigmoid(self.H_raw)  # Constrain to [0, 1]
        return self.H

    def forward(self, image_flat: torch.Tensor) -> torch.Tensor:
        """
        Compute bucket detector measurements: y = H · x_flat

        Args:
            image_flat: Flattened image, shape (B, N²).

        Returns:
            Bucket measurements, shape (B, M).
        """
        H = self.patterns
        return torch.matmul(image_flat, H.t())  # (B, N²) × (N², M) → (B, M)


class GhostImagingForwardModel(nn.Module):
    """
    Complete differentiable forward model for ghost imaging.

    Combines:
        1. PSF blur (Zernike-based defocus)
        2. Optional scattering degradation
        3. Pattern modulation (measurement matrix)
        4. Bucket detection (summation)
        5. Photon noise (Poisson + dark + readout)

    The differentiable version computes expected values for use with Poisson NLL loss.
    The sampling version generates realistic noisy data for training/testing.

    Args:
        image_size: Image resolution N.
        num_patterns: Number of illumination patterns M.
        psf_size: PSF kernel size.
        num_zernike: Number of Zernike coefficients.
        mean_photon_count: Mean photons per pixel.
        dark_count_rate: Dark counts per pixel.
        readout_noise_std: Readout noise std.
        pattern_type: Measurement pattern type.
    """

    def __init__(
        self,
        image_size: int = 64,
        num_patterns: int = 512,
        psf_size: int = 21,
        num_zernike: int = 11,
        mean_photon_count: float = 10.0,
        dark_count_rate: float = 0.5,
        readout_noise_std: float = 0.1,
        pattern_type: str = "learned",
        scattering_coeff: float = 0.0,
    ):
        super().__init__()

        self.image_size = image_size
        self.num_patterns = num_patterns

        # Sub-modules
        self.psf_generator = ZernikePSF(
            psf_size=psf_size,
            num_zernike=num_zernike,
        )
        self.measurement_matrix = MeasurementMatrix(
            num_patterns=num_patterns,
            image_size=image_size,
            pattern_type=pattern_type,
        )
        self.noise_model = PhotonNoiseModel(
            mean_photon_count=mean_photon_count,
            dark_count_rate=dark_count_rate,
            readout_noise_std=readout_noise_std,
        )
        self.scattering = ScatteringModel(scattering_coeff=scattering_coeff)

        self.mean_photon_count = mean_photon_count
        self.dark_count_rate = dark_count_rate

    def forward_expected(
        self,
        image: torch.Tensor,
        zernike_coeffs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the EXPECTED (noise-free) bucket measurements.

        Used in the Poisson NLL loss for physics-informed training:
            L_physics = Poisson_NLL(y_expected, y_measured)

        Args:
            image: Clean image, shape (B, 1, H, W).
            zernike_coeffs: Zernike coefficients, shape (B, num_zernike).

        Returns:
            Expected bucket measurements (in photon counts), shape (B, M).
        """
        # 1. Generate PSF and blur
        psf = self.psf_generator(zernike_coeffs)  # (B, 1, Kh, Kw)
        blurred = apply_psf_blur(image, psf)       # (B, 1, H, W)

        # 2. Scattering
        blurred = self.scattering(blurred)

        # 3. Flatten and measure
        B = blurred.shape[0]
        blurred_flat = blurred.view(B, -1)                # (B, N²)
        measurements = self.measurement_matrix(blurred_flat)  # (B, M)

        # 4. Scale to photon counts (expected value of Poisson)
        expected_counts = self.mean_photon_count * measurements + self.dark_count_rate

        return expected_counts.clamp(min=1e-6)  # Avoid log(0) in Poisson NLL

    def forward_sample(
        self,
        image: torch.Tensor,
        zernike_coeffs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate realistic noisy bucket measurements via Poisson sampling.

        Used for data generation (not differentiable through Poisson).

        Args:
            image: Clean image, shape (B, 1, H, W).
            zernike_coeffs: Zernike coefficients, shape (B, num_zernike).

        Returns:
            Tuple of:
                - Noisy bucket measurements (normalized), shape (B, M).
                - Expected bucket measurements (for reference), shape (B, M).
        """
        expected = self.forward_expected(image, zernike_coeffs)

        # Poisson sampling
        noisy_counts = torch.poisson(expected)

        # Add readout noise
        readout = torch.randn_like(noisy_counts) * self.noise_model.readout_noise_std
        noisy_counts = (noisy_counts + readout).clamp(min=0.0)

        # Normalize
        noisy_normalized = noisy_counts / (self.mean_photon_count + 1e-8)

        return noisy_normalized, expected

    def forward(
        self,
        image: torch.Tensor,
        zernike_coeffs: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        """
        Run the forward model.

        Args:
            image: Clean image, shape (B, 1, H, W).
            zernike_coeffs: Zernike coefficients, shape (B, num_zernike).
            sample: If True, return noisy samples; if False, return expected values.

        Returns:
            Bucket measurements, shape (B, M).
        """
        if sample:
            noisy, _ = self.forward_sample(image, zernike_coeffs)
            return noisy
        else:
            return self.forward_expected(image, zernike_coeffs)
