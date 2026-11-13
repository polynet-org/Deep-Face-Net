import time
import threading
from typing import Any, List, Optional

import core.config as config
import onnxruntime
import insightface

from core.config import (
    ANALYSIS_MODEL,
    LIVE_FACE_DETECTION_SIZE,
    FILE_FACE_DETECTION_SIZE,
    ADAPTIVE_DETECTION_ENABLED,
    ADAPTIVE_DETECTION_INTERVAL,
)
from download_models import check_model_status

# ---------------------------------------------------------------------------
# Singleton analyser (thread-safe double-checked locking)
# ---------------------------------------------------------------------------

_FACE_ANALYSER: Optional[Any] = None
_LOCK = threading.Lock()

# Separate analysers for live (fast) and offline (quality) modes
_LIVE_ANALYSER: Optional[Any] = None
_FILE_ANALYSER: Optional[Any] = None
_LIVE_LOCK = threading.Lock()
_FILE_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Adaptive detection cache (live-mode only)
# ---------------------------------------------------------------------------
_DETECTION_CACHE: dict = {}
_LAST_DETECTION_TIME: float = 0.0


def _ensure_model_downloaded():
    is_downloaded, _, _ = check_model_status(ANALYSIS_MODEL)
    if not is_downloaded:
        raise RuntimeError(
            f"The '{ANALYSIS_MODEL}' model is not downloaded. "
            "Please go to the Models tab to download it first."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_face_analyser(live_mode: bool = False) -> Any:
    """Return a thread-safe InsightFace analyser.

    Parameters
    ----------
    live_mode:
        When ``True`` a 320 × 320 detection window is used (faster, better
        for real-time).  When ``False`` a full 640 × 640 window is used for
        higher offline accuracy.
    """
    global _LIVE_ANALYSER, _FILE_ANALYSER

    if live_mode:
        if _LIVE_ANALYSER is None:
            with _LIVE_LOCK:
                if _LIVE_ANALYSER is None:
                    _ensure_model_downloaded()
                    analyser = insightface.app.FaceAnalysis(
                        name=ANALYSIS_MODEL,
                        allowed_modules=["detection", "recognition"],
                        providers=config.EXECUTION_PROVIDERS,
                    )
                    analyser.prepare(ctx_id=_context_id(), det_size=LIVE_FACE_DETECTION_SIZE)
                    _LIVE_ANALYSER = analyser
        return _LIVE_ANALYSER
    else:
        if _FILE_ANALYSER is None:
            with _FILE_LOCK:
                if _FILE_ANALYSER is None:
                    _ensure_model_downloaded()
                    analyser = insightface.app.FaceAnalysis(
                        name=ANALYSIS_MODEL,
                        allowed_modules=["detection", "recognition"],
                        providers=config.EXECUTION_PROVIDERS,
                    )
                    analyser.prepare(ctx_id=_context_id(), det_size=FILE_FACE_DETECTION_SIZE)
                    _FILE_ANALYSER = analyser
        return _FILE_ANALYSER


def get_one_face(frame, live_mode: bool = False) -> Optional[Any]:
    """Return the leftmost face detected in *frame*, or ``None``."""
    faces = get_face_analyser(live_mode).get(frame)
    try:
        return min(faces, key=lambda x: x.bbox[0])
    except (ValueError, TypeError):
        return None


def get_many_faces(frame, live_mode: bool = False) -> Optional[List[Any]]:
    """Return all faces detected in *frame*, or ``None`` if none found."""
    try:
        faces = get_face_analyser(live_mode).get(frame)
        return faces if faces else None
    except IndexError:
        return None


def get_faces_cached(
    frame,
    many: bool = False,
    use_cache: bool = True,
) -> Optional[List[Any]]:
    """Live-mode face detection with an adaptive time-based cache.

    If ``use_cache`` is true and the last detection result is younger than
    ``ADAPTIVE_DETECTION_INTERVAL`` seconds, the cached result is returned
    without re-running the model — giving a significant FPS boost.

    Parameters
    ----------
    frame:
        BGR numpy array (current webcam frame).
    many:
        If ``True`` return all faces; otherwise return only the best one.
    use_cache:
        Disable to force a fresh detection (e.g. first frame).
    """
    global _LAST_DETECTION_TIME, _DETECTION_CACHE

    now = time.monotonic()
    if (
        ADAPTIVE_DETECTION_ENABLED
        and use_cache
        and _DETECTION_CACHE
        and (now - _LAST_DETECTION_TIME) < ADAPTIVE_DETECTION_INTERVAL
    ):
        return _DETECTION_CACHE.get("faces")

    # Run detection
    if many:
        faces = get_many_faces(frame, live_mode=True)
    else:
        face = get_one_face(frame, live_mode=True)
        faces = [face] if face else None

    _LAST_DETECTION_TIME = now
    _DETECTION_CACHE["faces"] = faces
    return faces


def reset_detection_cache() -> None:
    """Clear the live-mode detection cache (call when switching sources)."""
    global _DETECTION_CACHE, _LAST_DETECTION_TIME
    _DETECTION_CACHE = {}
    _LAST_DETECTION_TIME = 0.0


_LANDMARK_MODEL = None
_LANDMARK_LOCK = threading.Lock()


def _context_id():
    return 0 if ("CUDAExecutionProvider" in config.EXECUTION_PROVIDERS
                 and "CUDAExecutionProvider" in onnxruntime.get_available_providers()) else -1


def ensure_face_landmarks(frame, face):
    """Run 106-point alignment only for faces needing masks, in either mode."""
    global _LANDMARK_MODEL
    if getattr(face, "landmark_2d_106", None) is not None:
        return
    with _LANDMARK_LOCK:
        if _LANDMARK_MODEL is None:
            path = config.BUFFALO_L_DIR / "2d106det.onnx"
            if not path.is_file():
                raise FileNotFoundError(f"Masking requires the landmark model: {path}")
            model = insightface.model_zoo.get_model(
                str(path), providers=config.EXECUTION_PROVIDERS)
            model.prepare(ctx_id=_context_id())
            _LANDMARK_MODEL = model
        _LANDMARK_MODEL.get(frame, face)
