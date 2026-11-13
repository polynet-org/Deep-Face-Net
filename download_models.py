import os
import requests
from tqdm import tqdm
import argparse
import sys
from pathlib import Path

try:
    from core import config

    CONFIG_LOADED = True
except ImportError:
    CONFIG_LOADED = False

MODELS = {
    "inswapper_128.onnx": {
        "url": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        "size": 554259814,
        "type": "swapper",
        "required": True,
        "description": "Face swap model (FP32)",
        "location": "models",
    },
    "inswapper_128.fp16.onnx": {
        "url": "https://huggingface.co/netrunner-exe/Insight-Swap-models/resolve/main/inswapper_128.fp16.onnx",
        "size": 277129907,
        "type": "swapper",
        "required": False,
        "description": "Face swap model (FP16, lighter)",
        "location": "models",
    },
    "GFPGANv1.4.onnx": {
        "url": "https://huggingface.co/neurobytemind/GFPGANv1.4.onnx/resolve/main/GFPGANv1.4.onnx",
        "size": 0,
        "type": "enhancer",
        "required": False,
        "description": "Face enhancement model (ONNX)",
        "location": "models",
    },
    "gfpgan-1024.onnx": {
        "url": "https://huggingface.co/neurobytemind/GFPGANv1.4.onnx/resolve/main/gfpgan-1024.onnx",
        "size": 0,
        "type": "enhancer",
        "required": False,
        "description": "Face enhancement HD model (ONNX)",
        "location": "models",
    },
    "GPEN-BFR-256.onnx": {
        "url": "https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/GPEN-BFR/GPEN-BFR-256.onnx",
        "size": 0,
        "type": "enhancer",
        "required": False,
        "description": "GPEN-BFR-256 Face Restoration (ONNX)",
        "location": "models",
    },
    "GPEN-BFR-512.onnx": {
        "url": "https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/GPEN-BFR/GPEN-BFR-512.onnx",
        "size": 0,
        "type": "enhancer",
        "required": False,
        "description": "GPEN-BFR-512 Face Restoration (ONNX)",
        "location": "models",
    },
    "buffalo_l": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "size": 326587068,
        "type": "analyser",
        "required": True,
        "description": "Face detection & analysis pack",
        "location": "insightface",
        "expected_files": [
            "1k3d68.onnx",
            "2d106det.onnx",
            "det_10g.onnx",
            "genderage.onnx",
            "w600k_r50.onnx",
        ],
    },
}


def get_model_path(model_name):
    """Get the full path where a model should be located."""
    info = MODELS.get(model_name)
    if not info:
        return None

    if info["location"] == "insightface":
        return Path.home() / ".insightface" / "models" / model_name
    else:
        # Check if model already exists in local source folder
        if CONFIG_LOADED and hasattr(config, "BASE_DIR"):
            local_p = config.BASE_DIR / "models" / model_name
            if local_p.exists():
                return local_p
        base = config.MODELS_DIR if CONFIG_LOADED else (Path.home() / ".deepfacenet" / "models")
        return base / model_name


def check_model_status(model_name):
    """Check if a model is downloaded. Returns (is_downloaded, path, current_size)."""
    info = MODELS.get(model_name)
    if not info:
        return False, None, 0

    path = get_model_path(model_name)

    if info["location"] == "insightface":
        # Check if directory exists with expected files
        expected = info.get("expected_files", [])
        if path.is_dir() and all((path / f).exists() for f in expected):
            total_size = sum((path / f).stat().st_size for f in expected)
            return True, path, total_size
        return False, path, 0
    else:
        if path and path.exists():
            return True, path, path.stat().st_size
        if CONFIG_LOADED and hasattr(config, "BASE_DIR"):
            fallback = config.BASE_DIR / "models" / model_name
            if fallback.exists():
                return True, fallback, fallback.stat().st_size
        return False, path, 0


def format_size(size_bytes):
    """Format bytes to human readable string."""
    if size_bytes == 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def download_file(url, filename, expected_size=None):
    if not url:
        print(f"Skipping {filename}: No download URL available.")
        return

    local_path = get_model_path(filename)
    if not local_path:
        base = config.MODELS_DIR if CONFIG_LOADED else (Path.home() / ".deepfacenet" / "models")
        local_path = base / filename

    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path_str = str(local_path)

    if local_path.exists():
        # Check size if available
        if expected_size:
            local_size = local_path.stat().st_size
            # Allow some tolerance for size mismatch (e.g. 5%)
            if abs(local_size - expected_size) / expected_size < 0.05:
                print(f"{filename} already exists and size matches. Skipping.")
                return
            else:
                print(
                    f"{filename} exists but size mismatch ({local_size} vs {expected_size}). Redownloading..."
                )
        else:
            print(f"{filename} already exists. Skipping.")
            return

    print(f"Downloading {filename} to {local_path_str}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte

        with (
            open(local_path_str, "wb") as f,
            tqdm(
                desc=filename,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        if local_path.exists():
            local_path.unlink()

def conditional_download(target_dir, urls):
    """Download multiple URLs into target_dir if they differ in name."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for url in urls:
        filename = url.split('/')[-1]
        download_file(url, filename)


def get_required_models(args):
    """Determine which models to download based on config and args."""

    if args.model:
        return [m for m in MODELS if m in args.model]

    if args.all:
        return list(MODELS.keys())

    required = []
    swapper_model = "inswapper_128.onnx"

    if CONFIG_LOADED:
        try:
            configured_swapper = config.SWAPPER_MODEL.name
            if configured_swapper in MODELS:
                swapper_model = configured_swapper
            elif "fp16" in configured_swapper and "inswapper_128.fp16.onnx" in MODELS:
                swapper_model = "inswapper_128.fp16.onnx"
            else:
                print(
                    f"Note: Configured swapper '{configured_swapper}' not in download list. Using default."
                )

        except Exception as e:
            print(f"Error reading config: {e}")

    if swapper_model not in required:
        required.append(swapper_model)

    return required


def main():
    parser = argparse.ArgumentParser(
        description="Download required models for Deep-Face-Net"
    )
    parser.add_argument(
        "--all", action="store_true", help="Download all available models"
    )
    parser.add_argument("--model", nargs="+", help="Download specific model(s) by name")
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for m in MODELS:
            print(f" - {m}")
        return

    models_dir = config.MODELS_DIR if CONFIG_LOADED else (Path.home() / ".deepfacenet" / "models")
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Models directory: {models_dir}")

    models_to_download = get_required_models(args)

    print(f"Targets: {', '.join(models_to_download)}")

    for filename in models_to_download:
        if filename in MODELS:
            data = MODELS[filename]
            download_file(data["url"], filename, data.get("size"))
        else:
            print(f"Warning: Model {filename} not defined in script.")

    print("\nDownload process completed.")


if __name__ == "__main__":
    main()

