"""
Single-Photon Noise Model for SPAD Detectors.

Models the quantum statistics of photon detection:
    y = Poisson(signal + dark_counts) + N(0, σ²_readout)

References:
    - Fossum, E.R. (2013). "Modeling of the quanta image sensor."
    - Gnanasambandam et al. (2020). "Image classification in the dark using quanta image sensors."
"""

import torch
import torch.nn as nn
import numpy as np


class PhotonNoiseModel(nn.Module):
    """
    Differentiable (forward-pass) single-photon noise model.

    The noise pipeline:
        1. Scale clean image by mean photon count: λ = N̄ · x_clean
        2. Add dark current: λ_total = λ + d
        3. Sample from Poisson(λ_total)
        4. Add Gaussian readout noise: y = Poisson(λ_total) + N(0, σ²)
        5. Normalize back to [0, 1] range

    Note: Poisson sampling is NOT differentiable. For the differentiable
    forward model used in training, we use the Poisson NLL loss instead.
    This module is used for DATA GENERATION only.

    Args:
        mean_photon_count: Average number of photons per pixel.
        dark_count_rate: Mean dark counts per pixel per frame.
        readout_noise_std: Standard deviation of Gaussian readout noise.
    """

    def __init__(
        self,
        mean_photon_count: float = 10.0,
        dark_count_rate: float = 0.5,
        readout_noise_std: float = 0.1,
    ):
        super().__init__()
        self.mean_photon_count = mean_photon_count
        self.dark_count_rate = dark_count_rate
        self.readout_noise_std = readout_noise_std

    def forward(self, clean_signal: torch.Tensor) -> torch.Tensor:
        """
        Apply single-photon noise to a clean signal.

        Args:
            clean_signal: Clean intensity values in [0, 1], any shape.

        Returns:
            Noisy signal (normalized back to approximate [0, 1] range).
        """
        # Scale to photon counts
        lam = self.mean_photon_count * clean_signal.clamp(min=0.0)

        # Add dark current
        lam_total = lam + self.dark_count_rate

        # Poisson sampling (not differentiable — used for data generation)
        poisson_counts = torch.poisson(lam_total)

        # Readout noise
        if self.readout_noise_std > 0:
            readout = torch.randn_like(poisson_counts) * self.readout_noise_std
            noisy_counts = poisson_counts + readout
        else:
            noisy_counts = poisson_counts

        # Normalize back
        noisy_normalized = noisy_counts / (self.mean_photon_count + 1e-8)

        return noisy_normalized.clamp(min=0.0)

    def get_expected_snr(self) -> float:
        """
        Compute the expected signal-to-noise ratio for a mid-gray pixel (x=0.5).

        SNR = N̄·x / sqrt(N̄·x + d + σ²_read)
        """
        signal = self.mean_photon_count * 0.5
        noise_var = signal + self.dark_count_rate + self.readout_noise_std ** 2
        return signal / np.sqrt(noise_var)


class ScatteringModel(nn.Module):
    """
    Atmospheric / media scattering degradation model.

    Implements the Koschmieder model:
        I_scattered = I_direct · exp(-τ·d) + A · (1 - exp(-τ·d))

    where:
        τ = scattering coefficient
        d = optical depth (normalized to 1 for simplicity)
        A = ambient/airlight intensity (typically 1.0 for fog)

    Args:
        scattering_coeff: Scattering coefficient τ.
        airlight: Ambient airlight intensity A.
    """

    def __init__(self, scattering_coeff: float = 0.0, airlight: float = 1.0):
        super().__init__()
        self.scattering_coeff = scattering_coeff
        self.airlight = airlight

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply scattering degradation to an image.

        Args:
            image: Input image in [0, 1], shape (B, C, H, W) or (C, H, W).

        Returns:
            Scattered image.
        """
        if self.scattering_coeff <= 0:
            return image

        transmission = np.exp(-self.scattering_coeff)
        scattered = image * transmission + self.airlight * (1.0 - transmission)
        return scattered.clamp(0.0, 1.0)
