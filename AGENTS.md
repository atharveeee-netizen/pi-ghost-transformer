# Antigravity AI Lab Instructions

If the user requests to start training or run the lab training, follow these exact steps:

1. **Environment Setup:** Ensure `torch`, `torchvision`, `scipy`, and `numpy` are installed. If missing, install them. Since this is the AI lab, default to installing the CUDA versions (e.g., `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` or as appropriate for the lab's GPU architecture).
2. **Dataset Configuration:** Remind the user to set the dataset paths in `config.py` (e.g., `mvtec_path`). If they don't have the dataset yet, they can use the `SyntheticFabricGenerator` which is default.
3. **Execution:** Execute the full ablation matrix required for IEEE submission by running:
   ```bash
   python ablations.py
   ```
4. **Monitoring:** Allow the task to run in the background. The ablation script will output results to `results/ablation_results.json`. Notify the user when the final JSON is ready.

This project is a **Physics-Informed Ghost Transformer (PI-GhostFormer)** for single-photon computational imaging. It uses a custom **Poisson NLL Loss**, a **Differentiable Zernike Wavefront Layer**, and a **64-Token Spatial Physics-Conditioned Cross-Attention (PCCA)** decoder. Do NOT modify these core mathematical components unless explicitly requested, as they are specifically tuned for a Q1 journal submission.
