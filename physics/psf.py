"""
Zernike Polynomial-Based Point Spread Function (PSF) Generator.

Implements the wave-optics PSF model using Zernike polynomial wavefront expansion.
The PSF is computed as:

    PSF(r) = |FT{ P(ρ) · exp(iW(ρ)) }|²

where:
    P(ρ) = circular pupil function
    W(ρ) = Σ aₙ · Zₙ(ρ)  — wavefront aberration (Zernike expansion)
    a₄   = defocus coefficient (primary variable in this study)

References:
    - Noll, R.J. (1976). "Zernike polynomials and atmospheric turbulence." JOSA.
    - Goodman, J.W. (2005). "Introduction to Fourier Optics." 3rd ed.
"""

import math

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def _zernike_radial(n: int, m: int, rho: torch.Tensor) -> torch.Tensor:
    """
    Compute the radial component R_n^m(ρ) of the Zernike polynomial.

    Uses the closed-form summation:
        R_n^m(ρ) = Σ_{s=0}^{(n-m)/2} (-1)^s * (n-s)! / [s! * ((n+m)/2-s)! * ((n-m)/2-s)!] * ρ^(n-2s)
    """
    m_abs = abs(m)
    if (n - m_abs) % 2 != 0:
        return torch.zeros_like(rho)

    result = torch.zeros_like(rho)
    for s in range((n - m_abs) // 2 + 1):
        coeff = ((-1) ** s * math.factorial(n - s)) / (
            math.factorial(s)
            * math.factorial((n + m_abs) // 2 - s)
            * math.factorial((n - m_abs) // 2 - s)
        )
        result = result + coeff * rho ** (n - 2 * s)
    return result


def zernike_polynomial(n: int, m: int, rho: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Compute the Zernike polynomial Z_n^m(ρ, θ) in standard (non-Noll) indexing.

    Args:
        n: Radial order.
        m: Azimuthal frequency (can be negative).
        rho: Radial coordinate tensor (normalized to [0, 1]).
        theta: Angular coordinate tensor [radians].

    Returns:
        Z_n^m(ρ, θ) evaluated at each point.
    """
    R = _zernike_radial(n, m, rho)
    if m > 0:
        return R * torch.cos(m * theta)
    elif m < 0:
        return R * torch.sin(-m * theta)
    else:
        return R


# Noll-indexed Zernike modes: (n, m) pairs for indices 1..15
NOLL_TO_NM = {
    1: (0, 0),    # Piston
    2: (1, 1),    # Tip (x-tilt)
    3: (1, -1),   # Tilt (y-tilt)
    4: (2, 0),    # Defocus  ← PRIMARY VARIABLE
    5: (2, -2),   # Oblique astigmatism
    6: (2, 2),    # Vertical astigmatism
    7: (3, -1),   # Vertical coma
    8: (3, 1),    # Horizontal coma
    9: (3, -3),   # Vertical trefoil
    10: (3, 3),   # Oblique trefoil
    11: (4, 0),   # Primary spherical
}


class ZernikePSF(torch.nn.Module):
    """
    Differentiable PSF generator based on Zernike wavefront expansion.

    The wavefront aberration W(ρ, θ) is expanded as:
        W(ρ, θ) = Σ_{j=1}^{J} aⱼ · Zⱼ(ρ, θ)

    The complex pupil function is:
        P_complex(ρ, θ) = P(ρ) · exp(i · 2π · W(ρ, θ))

    The PSF is the squared modulus of the Fourier transform:
        PSF = |FT{P_complex}|²

    Args:
        psf_size: Output PSF kernel size (odd).
        num_zernike: Number of Zernike coefficients (Noll indexing).
        wavelength: Illumination wavelength [m].
        pupil_samples: Number of samples across the pupil diameter.
    """

    def __init__(
        self,
        psf_size: int = 21,
        num_zernike: int = 11,
        wavelength: float = 550e-9,
        pupil_samples: int = 128,
    ):
        super().__init__()
        self.psf_size = psf_size
        self.num_zernike = num_zernike
        self.wavelength = wavelength
        self.pupil_samples = pupil_samples

        # Pre-compute pupil coordinates
        x = torch.linspace(-1, 1, pupil_samples)
        y = torch.linspace(-1, 1, pupil_samples)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        rho = torch.sqrt(xx**2 + yy**2)
        theta = torch.atan2(yy, xx)

        # Circular pupil mask
        pupil_mask = (rho <= 1.0).float()

        self.register_buffer("rho", rho)
        self.register_buffer("theta", theta)
        self.register_buffer("pupil_mask", pupil_mask)

        # Pre-compute Zernike basis functions (fixed, not learned)
        basis = []
        for j in range(1, num_zernike + 1):
            if j in NOLL_TO_NM:
                n, m = NOLL_TO_NM[j]
                Z_j = zernike_polynomial(n, m, rho, theta)
                basis.append(Z_j)
            else:
                basis.append(torch.zeros_like(rho))

        # Shape: (num_zernike, pupil_samples, pupil_samples)
        self.register_buffer("zernike_basis", torch.stack(basis, dim=0))

    def compute_wavefront(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Compute the wavefront aberration W(ρ, θ) from Zernike coefficients.

        Args:
            coefficients: Zernike coefficients, shape (B, num_zernike) or (num_zernike,).

        Returns:
            Wavefront map, shape (B, pupil_samples, pupil_samples) or (pupil_samples, pupil_samples).
        """
        if coefficients.dim() == 1:
            # (num_zernike,) × (num_zernike, H, W) → (H, W)
            wavefront = torch.einsum("j,jhw->hw", coefficients, self.zernike_basis)
        else:
            # (B, num_zernike) × (num_zernike, H, W) → (B, H, W)
            wavefront = torch.einsum("bj,jhw->bhw", coefficients, self.zernike_basis)
        return wavefront

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Generate the PSF for given Zernike coefficients.

        Args:
            coefficients: Zernike coefficients, shape (B, num_zernike) or (num_zernike,).

        Returns:
            Normalized PSF kernel, shape (B, 1, psf_size, psf_size) or (1, psf_size, psf_size).
        """
        wavefront = self.compute_wavefront(coefficients)
        single = wavefront.dim() == 2

        if single:
            wavefront = wavefront.unsqueeze(0)

        # Complex pupil function: P(ρ) · exp(i · 2π · W(ρ,θ))
        phase = 2.0 * np.pi * wavefront
        pupil_complex = self.pupil_mask.unsqueeze(0) * torch.exp(1j * phase)

        # PSF = |FT{pupil_complex}|²
        ft = torch.fft.fftshift(torch.fft.fft2(pupil_complex), dim=(-2, -1))
        psf_full = torch.abs(ft) ** 2

        # Crop center to psf_size × psf_size
        center = self.pupil_samples // 2
        half = self.psf_size // 2
        psf_cropped = psf_full[
            :,
            center - half : center + half + 1,
            center - half : center + half + 1,
        ]

        # Normalize so PSF sums to 1 (energy conservation)
        psf_cropped = psf_cropped / (psf_cropped.sum(dim=(-2, -1), keepdim=True) + 1e-10)

        # Add channel dimension: (B, 1, H, W)
        psf_cropped = psf_cropped.unsqueeze(1).float()

        if single:
            return psf_cropped.squeeze(0)
        return psf_cropped

    def get_defocus_psf(self, defocus_coeff: float) -> torch.Tensor:
        """
        Convenience method: generate a PSF with only defocus aberration (a₄).

        Args:
            defocus_coeff: Defocus Zernike coefficient (a₄) in waves.

        Returns:
            PSF kernel, shape (1, psf_size, psf_size).
        """
        coeffs = torch.zeros(self.num_zernike, device=self.rho.device)
        coeffs[3] = defocus_coeff  # Noll index 4 → array index 3
        return self.forward(coeffs)


def apply_psf_blur(image: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
    """
    Apply PSF blur to an image via convolution.

    This is the deterministic part of the forward degradation model:
        I_blurred = image ⊛ PSF

    Args:
        image: Input image, shape (B, 1, H, W) or (1, H, W).
        psf: PSF kernel, shape (B, 1, Kh, Kw) or (1, Kh, Kw).

    Returns:
        Blurred image, same shape as input.
    """
    single = image.dim() == 3
    if single:
        image = image.unsqueeze(0)
    if psf.dim() == 3:
        psf = psf.unsqueeze(0)

    B = image.shape[0]
    pad_h = psf.shape[2] // 2
    pad_w = psf.shape[3] // 2

    # Per-sample convolution (each sample may have a different PSF)
    if psf.shape[0] == 1 and B > 1:
        # Shared PSF across batch
        image_padded = F.pad(image, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        blurred = F.conv2d(image_padded, psf.flip(-1, -2), padding=0)
    elif psf.shape[0] == B:
        # Per-sample PSF: use groups convolution trick
        blurred_list = []
        for i in range(B):
            img_i = image[i : i + 1]
            psf_i = psf[i : i + 1]
            img_padded = F.pad(img_i, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
            blurred_i = F.conv2d(img_padded, psf_i.flip(-1, -2), padding=0)
            blurred_list.append(blurred_i)
        blurred = torch.cat(blurred_list, dim=0)
    else:
        raise ValueError(f"PSF batch size {psf.shape[0]} incompatible with image batch size {B}")

    blurred = torch.clamp(blurred, min=0.0)

    if single:
        return blurred.squeeze(0)
    return blurred
