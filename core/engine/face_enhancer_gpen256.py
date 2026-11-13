"""
core/engine/face_enhancer_gpen256.py
====================================
GPEN-BFR-256 ONNX-based Face Restoration.
"""

from typing import List, Optional

import cv2
import numpy as np

import core.config as config
from core.engine._onnx_enhancer import (
    create_onnx_session,
    enhance_face_onnx,
    warmup_session,
)

_ENHANCER_SESSION = None
_INPUT_SIZE = 256
_MODEL_URL = "https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/GPEN-BFR/GPEN-BFR-256.onnx"

def get_enhancer_session():
    global _ENHANCER_SESSION
    if _ENHANCER_SESSION is None:
        model_path = str(config.GPEN_256_MODEL)
        if not config.GPEN_256_MODEL.exists():
            from download_models import conditional_download
            print(f"Downloading GPEN-256 from {_MODEL_URL}...")
            conditional_download(str(config.MODELS_DIR), [_MODEL_URL])

        _ENHANCER_SESSION = create_onnx_session(model_path)
        warmup_session(_ENHANCER_SESSION)
    return _ENHANCER_SESSION

def enhance_faces(frame: np.ndarray, faces: List[Any], frame_size: tuple) -> np.ndarray:
    """Enhance multiple faces in a single frame using the GPEN-256 model.
    Matches the signature expected by video_processor.py and deepfake_app.py."""
    if not faces:
        return frame

    result_frame = frame.copy()
    try:
        session = get_enhancer_session()
        for face in faces:
            result_frame = enhance_face_onnx(result_frame, face, session, _INPUT_SIZE)
    except Exception as e:
        print(f"Error during GPEN-256 enhancement: {e}")

    return result_frame
