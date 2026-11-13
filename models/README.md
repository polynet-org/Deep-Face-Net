# Models Directory

This directory contains the model files required for Deep-Face-Net.

Due to GitHub's file size limits, large model files are not included in the repository. You can download them automatically using the provided script or manually from the links below.

## Automatic Download

Run the following command from the root content of the repository:

```bash
python download_models.py
```

This will download the core models required for face swapping.

## Manual Download

If the script fails, please download the following files and place them in this directory:

| File Name | Description | Size | Source |
|-----------|-------------|------|--------|
| `inswapper_128.onnx` | Face swap model (FP32) | ~554MB | [Download](https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx) |
| `inswapper_128.fp16.onnx` | Face swap model (FP16) | ~277MB | [Download](https://huggingface.co/netrunner-exe/Insight-Swap-models/resolve/main/inswapper_128.fp16.onnx) |
| `GFPGANv1.4.pth` | Face restoration model | ~350MB | [Download](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth) |

> **Note**: `inswapper_128.fp16.onnx` is an optimized version. If you configured the app to use it, ensure you assign it correctly in `core/config.py`.
