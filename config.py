"""
PI-GhostFormer: Physics-Informed Ghost Transformer
Central Configuration — All physics constants, model hyperparameters, and experiment settings.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PhysicsConfig:
    """Optical physics and sensor parameters."""

    # --- Image / Object Plane ---
    image_size: int = 64                    # Reconstructed image resolution (N×N)
    wavelength: float = 550e-9              # Illumination wavelength [m] (green light)
    pixel_pitch: float = 6.5e-6            # Detector pixel pitch [m]

    # --- Lens / PSF ---
    focal_length: float = 50e-3            # Lens focal length [m]
    aperture_diameter: float = 25e-3       # Lens aperture diameter [m]
    f_number: float = 2.0                  # f/# = focal_length / aperture_diameter
    num_zernike: int = 11                  # Number of Zernike coefficients (Noll indexing)
    psf_size: int = 21                     # PSF kernel size (odd number)

    # --- Defocus Range ---
    defocus_range: Tuple[float, float] = (0.0, 3.0)   # Zernike a₄ defocus coeff range [waves]
    num_defocus_levels: int = 7             # Number of discrete defocus levels for evaluation

    # --- Single-Photon / SPAD Sensor ---
    mean_photon_counts: List[float] = field(
        default_factory=lambda: [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    )
    default_photon_count: float = 10.0     # Default mean photons per pixel
    dark_count_rate: float = 0.5           # Dark counts per pixel per frame
    readout_noise_std: float = 0.1         # Readout noise standard deviation [counts]

    # --- Ghost Imaging / Single-Pixel ---
    num_patterns: int = 512                # Number of illumination patterns (M)
    sampling_ratios: List[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.1, 0.25, 0.5]
    )
    default_sampling_ratio: float = 0.125  # Default: M / N² = 512 / 4096 ≈ 12.5%
    pattern_type: str = "learned"          # "hadamard", "random", "learned"

    # --- Scattering (Bonus Experiment) ---
    scattering_coefficients: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.3, 0.5, 1.0]
    )


@dataclass
class ModelConfig:
    """PI-GhostFormer architecture hyperparameters."""

    # --- Encoder (1D bucket signal) ---
    embed_dim: int = 256                   # Transformer embedding dimension
    encoder_layers: int = 6                # Number of encoder Transformer blocks
    encoder_heads: int = 8                 # Multi-head attention heads (encoder)
    encoder_ff_dim: int = 1024             # Feed-forward hidden dimension
    dropout: float = 0.1                   # Dropout rate

    # --- CNN Stem (1D) ---
    stem_channels: List[int] = field(
        default_factory=lambda: [64, 128, 256]
    )
    stem_kernel_size: int = 7              # 1D convolution kernel size

    # --- Physics Token ---
    physics_embed_dim: int = 64            # PSF / Zernike embedding dimension
    physics_mlp_hidden: int = 128          # MLP hidden dim for physics embedding

    # --- Decoder (2D U-Net + Swin) ---
    decoder_channels: List[int] = field(
        default_factory=lambda: [256, 128, 64, 32]
    )
    swin_window_size: int = 4              # Swin Transformer window size
    swin_heads: List[int] = field(
        default_factory=lambda: [8, 4, 2, 1]
    )
    decoder_layers_per_stage: int = 2      # Swin blocks per decoder stage

    # --- Classification Head ---
    num_defect_classes: int = 5            # {Normal, Hole, Stain, Thread_break, Misweave}
    cls_hidden_dim: int = 256              # Classification MLP hidden dim

    # --- Learned Measurement Matrix ---
    learn_patterns: bool = True            # Co-optimize illumination patterns


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # --- Optimizer ---
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    warmup_epochs: int = 5
    max_epochs: int = 200
    batch_size: int = 16
    num_workers: int = 4

    # --- Loss Weights ---
    lambda_recon: float = 1.0              # MSE reconstruction loss weight
    lambda_physics: float = 0.5            # Poisson NLL physics-consistency weight
    lambda_perceptual: float = 0.1         # VGG perceptual loss weight
    lambda_classify: float = 0.3           # Classification cross-entropy weight
    lambda_tv: float = 0.01               # Total Variation regularizer weight

    # --- Scheduler ---
    scheduler: str = "cosine"              # "cosine", "step", "plateau"
    step_size: int = 50
    gamma: float = 0.5

    # --- Self-Supervised Mode ---
    self_supervised: bool = False           # Train without ground-truth images

    # --- Checkpointing ---
    save_every: int = 10                   # Save checkpoint every N epochs
    log_every: int = 5                     # Log metrics every N batches


@dataclass
class DataConfig:
    """Dataset and simulation parameters."""

    # --- Fabric Textures ---
    num_train_samples: int = 5000          # Number of training samples
    num_val_samples: int = 1000            # Number of validation samples
    num_test_samples: int = 1000           # Number of test samples

    # --- Defect Generation ---
    defect_probability: float = 0.5        # Probability of a sample having a defect
    defect_types: List[str] = field(
        default_factory=lambda: ["hole", "stain", "thread_break", "misweave"]
    )
    defect_size_range: Tuple[int, int] = (3, 15)   # Defect size in pixels

    # --- External Datasets ---
    aitex_path: str = ""                   # Path to AITEX dataset (set if available)
    mvtec_path: str = ""                   # Path to MVTec AD dataset (set if available)
    tilda_path: str = ""                   # Path to TILDA dataset (set if available)

    # --- Augmentation ---
    random_flip: bool = True
    random_rotation: bool = True
    rotation_degrees: float = 15.0


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # --- Experiment Meta ---
    experiment_name: str = "pi_ghostformer_v1"
    device: str = "cuda"                   # "cuda" or "cpu"
    seed: int = 42
    results_dir: str = "results"


def get_config() -> ExperimentConfig:
    """Returns the default experiment configuration."""
    return ExperimentConfig()
