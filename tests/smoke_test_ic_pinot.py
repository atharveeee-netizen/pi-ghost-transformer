import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

"""
End-to-End Verification Test for IC-PINOT:
In-Context Physics-Informed Neural Operator Transformer.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ic_pinot import ICPINOT
from models.evidential import EvidentialNIGLoss
from physics.fno import OpticalFourierNeuralOperator


def main():
    print("=" * 65)
    print("IC-PINOT END-TO-END PIPELINE VERIFICATION")
    print("=" * 65)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")
    
    # 1. Initialize IC-PINOT
    model = ICPINOT(
        num_patterns=512,
        embed_dim=256,
        encoder_layers=4,
        encoder_heads=8,
        physics_dim=11,
        img_size=64,
        num_classes=2,
        num_prompt_tokens=16
    ).to(device)
    
    criterion = EvidentialNIGLoss(coeff_reg=0.05).to(device)
    fno = OpticalFourierNeuralOperator(modes=12, width=32, physics_dim=11).to(device)
    
    # Dummy data
    B = 2
    y_q = torch.randn(B, 512, device=device)
    theta = torch.randn(B, 11, device=device)
    target_img = torch.rand(B, 1, 64, 64, device=device)
    
    # Test 1: Zero-shot standard pass
    print("\n[Test 1/4] Zero-Shot Standard Forward Pass (K=0)...")
    recon_mean, logits = model(y_q, theta)
    assert recon_mean.shape == (B, 1, 64, 64), f"Unexpected shape {recon_mean.shape}"
    assert logits.shape == (B, 2), f"Unexpected logits shape {logits.shape}"
    print(f"  Reconstruction output shape: {recon_mean.shape}")
    print(f"  Classification logits shape: {logits.shape}")
    print("  --> PASS")
    
    # Test 2: In-Context Adaptation Pass (K=3 context pairs)
    print("\n[Test 2/4] In-Context Adaptation Pass (K=3 prompt pairs)...")
    K = 3
    c_meas = torch.randn(B, K, 512, device=device)
    c_imgs = torch.rand(B, K, 1, 64, 64, device=device)
    recon_ic, logits_ic = model(y_q, theta, context_meas=c_meas, context_imgs=c_imgs)
    assert recon_ic.shape == (B, 1, 64, 64)
    print(f"  In-Context Adapted Reconstruction: {recon_ic.shape}")
    print("  --> PASS")
    
    # Test 3: Evidential Normal-Inverse-Gamma (NIG) Single-Pass Uncertainty
    print("\n[Test 3/4] Evidential NIG Single-Pass Calibrated Uncertainty...")
    nig_dict, logits_uq = model(y_q, theta, return_evidential=True)
    for k in ["gamma", "v", "alpha", "beta", "aleatoric", "epistemic", "total_uncertainty"]:
        assert k in nig_dict, f"Missing key {k}"
        assert nig_dict[k].shape == (B, 1, 64, 64)
    print(f"  Predicted Mean gamma:                   {nig_dict['gamma'].shape}")
    print(f"  Aleatoric Uncertainty beta/(alpha-1):   {nig_dict['aleatoric'].shape}")
    print(f"  Epistemic Uncertainty beta/(v(alpha-1)): {nig_dict['epistemic'].shape}")
    print(f"  Total Uncertainty:                      {nig_dict['total_uncertainty'].shape}")
    
    losses = criterion(nig_dict, target_img)
    print(f"  Evidential NLL Loss: {losses['loss_nll'].item():.4f} | Reg: {losses['loss_reg'].item():.4f}")
    losses['loss_evidential'].backward()
    print("  Backpropagation gradient flow: OK")
    print("  --> PASS")
    
    # Test 4: Fourier Neural Operator (FNO)
    print("\n[Test 4/4] Differentiable Optical Fourier Neural Operator...")
    fno_out = fno(target_img, theta)
    assert fno_out.shape == (B, 1, 64, 64)
    fno_loss = fno_out.sum()
    fno_loss.backward()
    print(f"  FNO continuous wave degradation field shape: {fno_out.shape}")
    print("  --> PASS")
    
    print("\n" + "=" * 65)
    print("ALL 4 IC-PINOT CORE ARCHITECTURAL TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    main()
