"""
Fourier Neural Operator (FNO) for Computational Optical Forward Modeling.
Implements Spectral Convolution in 2D Fourier space based on Li et al. (ICLR 2021).
Parameterizes continuous optical wave propagation and medium scattering directly in frequency space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    2D Fourier layer: FFT -> Linear Transform on lower modes -> IFFT.
    """
    def __init__(self, in_channels: int, out_channels: int, modes1: int = 12, modes2: int = 12):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        # Complex weights for the top frequency corners
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        # Compute 2D Fourier transform
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to spatial domain via Inverse Fourier Transform
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class OpticalFourierNeuralOperator(nn.Module):
    """
    Full Fourier Neural Operator mapping:
        (x_clean, optical_params_theta) -> x_degraded
    Capable of learning non-Zernike, vibration-induced, and media scattering degradations.
    """
    def __init__(self, modes: int = 12, width: int = 32, physics_dim: int = 11):
        super().__init__()
        self.modes = modes
        self.width = width
        
        # Lift input + optical params to feature space
        self.p = nn.Conv2d(1 + physics_dim, width, kernel_size=1)
        
        # Spectral convolution layers
        self.conv0 = SpectralConv2d(width, width, modes, modes)
        self.conv1 = SpectralConv2d(width, width, modes, modes)
        
        self.w0 = nn.Conv2d(width, width, kernel_size=1)
        self.w1 = nn.Conv2d(width, width, kernel_size=1)
        
        # Project back to image domain
        self.q = nn.Sequential(
            nn.Conv2d(width, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, image: torch.Tensor, optical_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 1, H, W) clean input image
            optical_params: (B, physics_dim) optical state parameters
        Returns:
            degraded_field: (B, 1, H, W) physically degraded optical field
        """
        B, C, H, W = image.shape
        # Spatial expansion of optical parameters
        params_spatial = optical_params.view(B, -1, 1, 1).expand(-1, -1, H, W)
        x = torch.cat([image, params_spatial], dim=1)
        
        # FNO feature propagation
        x = self.p(x)
        
        x1 = self.conv0(x) + self.w0(x)
        x = F.gelu(x1)
        
        x2 = self.conv1(x) + self.w1(x)
        x = F.gelu(x2)
        
        degraded = self.q(x)
        return torch.sigmoid(degraded)
