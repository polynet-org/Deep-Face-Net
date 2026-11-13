from pathlib import Path
from typing import Literal, List
import os
import sys

# ---------------- Paths ----------------

BASE_DIR = Path(__file__).resolve().parent.parent

def _resolve_models_dir() -> Path:
    # 1. Custom environment variable override if set
    if "DEEPFACENET_MODELS_DIR" in os.environ:
        p = Path(os.environ["DEEPFACENET_MODELS_DIR"])
        p.mkdir(parents=True, exist_ok=True)
        return p

    # 2. If running from local source checkout and local ./models is writable, use it
    if not getattr(sys, "frozen", False):
        local_dir = BASE_DIR / "models"
        try:
            local_dir.mkdir(parents=True, exist_ok=True)
            test_file = local_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
            return local_dir
        except (OSError, PermissionError):
            pass

    # 3. For AppImage, Windows Program Files, macOS bundle, or any read-only location:
    # Use persistent user-writable application data directory: ~/.deepfacenet/models
    user_models_dir = Path.home() / ".deepfacenet" / "models"
    user_models_dir.mkdir(parents=True, exist_ok=True)
    return user_models_dir

MODELS_DIR = _resolve_models_dir()

def _resolve_model_path(filename: str) -> Path:
    """Find model in local directory if present, otherwise in user MODELS_DIR."""
    local_file = BASE_DIR / "models" / filename
    if local_file.exists():
        return local_file
    return MODELS_DIR / filename

ANALYSIS_MODEL = "buffalo_l"

# Use the standard swapper on CPU and CUDA. The bundled FP16 conversion
# does not preserve source identity reliably and must not be auto-selected.
SWAPPER_MODEL      = _resolve_model_path("inswapper_128.onnx")
SWAPPER_MODEL_FP16 = _resolve_model_path("inswapper_128.fp16.onnx")

# Enhancer models
ENHANCER_MODEL     = _resolve_model_path("GFPGANv1.4.onnx")    # legacy default
GFPGAN_1024_MODEL  = _resolve_model_path("gfpgan-1024.onnx")   # higher quality
GPEN_256_MODEL     = _resolve_model_path("GPEN-BFR-256.onnx")
GPEN_512_MODEL     = _resolve_model_path("GPEN-BFR-512.onnx")

ENHANCE_WEIGHT = 0.6  # blend strength: 0 = full enhancer, 1 = original face

INSIGHTFACE_DIR = Path.home() / ".insightface" / "models"
BUFFALO_L_DIR   = INSIGHTFACE_DIR / "buffalo_l"

# ---------------- Execution ----------------

# Providers tried in order; the first one available wins at runtime.
# Overridable at launch (GUI or CLI).
EXECUTION_PROVIDERS: List[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# ---------------- Camera Settings ----------------

DEFAULT_CAMERA_INDEX = 0
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FPS    = 30

# ---------------- Face Detection ----------------

# Use a smaller detection size in live/streaming mode for speed,
# and full 640×640 for offline file processing quality.
LIVE_FACE_DETECTION_SIZE = (320, 320)
FILE_FACE_DETECTION_SIZE = (640, 640)
FACE_DETECTION_SIZE      = FILE_FACE_DETECTION_SIZE   # back-compat alias

FACE_CONFIDENCE_THRESHOLD = 0.5

# Live-mode adaptive detection cache: skip re-detection if last result
# is fresher than this interval (seconds).  ~30 fps detection cadence.
ADAPTIVE_DETECTION_ENABLED  = True
ADAPTIVE_DETECTION_INTERVAL = 0.033   # seconds  ≈ 30 Hz

# ---------------- Multi-Face / Mapping ----------------

MANY_FACES = False   # Swap every detected face using the default source face
MAP_FACES  = False   # Use source↔target embedding map for selective swapping

# ---------------- Live Mode ----------------

SHOW_FPS    = False
LIVE_MIRROR = False  # Horizontally flip webcam preview

# ---------------- Enhancer Mode ----------------

# Which face enhancer to run after swapping.
# "none"    — no enhancement
# "gfpgan"  — GFPGANv1.4 / gfpgan-1024 (auto-select whichever is present)
# "gpen256" — GPEN-BFR-256 (faster, lighter)
# "gpen512" — GPEN-BFR-512 (higher res)
ENHANCER_MODE: Literal["none", "gfpgan", "gpen256", "gpen512"] = "none"

# ---------------- Masking / Swapping ----------------

OPACITY                = 1.0
MOUTH_MASK_ENABLED     = False
MOUTH_MASK_SIZE        = 0.0
POISSON_BLEND_ENABLED  = False
EYES_MASK_ENABLED      = False
EYES_MASK_SIZE         = 0.0
EYEBROWS_MASK_ENABLED  = False
EYEBROWS_MASK_SIZE     = 0.0
MASK_FEATHER_RATIO     = 10
MASK_DOWN_SIZE         = 1.0
SHARPNESS              = 0.0
ENABLE_INTERPOLATION   = False
INTERPOLATION_WEIGHT   = 0.2
