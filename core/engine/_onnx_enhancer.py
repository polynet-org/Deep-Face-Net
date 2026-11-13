"""
core/engine/_onnx_enhancer.py
==============================
Shared ONNX inference helpers for GPEN-style face-restoration models.

Ported from Deep-Live-Cam 2.7's modules/processors/frame/_onnx_enhancer.py.
Both GPEN-BFR-256 and GPEN-BFR-512 use this as their inference backend.
"""

import threading
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime

import core.config as config


# Standard FFHQ 5-point face template for 512×512 (scaled for other sizes)
_FFHQ_512 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19366],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

def create_onnx_session(model_path: str) -> onnxruntime.InferenceSession:
    """Load *model_path* into an ONNX Runtime session using the execution
    providers defined in ``core.config.EXECUTION_PROVIDERS``."""
    providers = getattr(config, "EXECUTION_PROVIDERS", ["CPUExecutionProvider"])
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    return onnxruntime.InferenceSession(model_path, sess_options=opts, providers=providers)


def warmup_session(session: onnxruntime.InferenceSession) -> None:
    """Run one dummy inference to JIT-compile kernels before live use."""
    inp = session.get_inputs()[0]
    h = inp.shape[2] if isinstance(inp.shape[2], int) else 256
    w = inp.shape[3] if isinstance(inp.shape[3], int) else 256
    dummy = np.zeros((1, 3, h, w), dtype=np.float32)
    try:
        session.run(None, {inp.name: dummy})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Alignment helpers (shared with face_enhancer.py)
# ---------------------------------------------------------------------------

def _align_face(
    frame: np.ndarray,
    landmarks_5: np.ndarray,
    output_size: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    scale = output_size / 512.0
    template = _FFHQ_512 * scale
    affine_matrix, _ = cv2.estimateAffinePartial2D(landmarks_5, template, method=cv2.LMEDS)
    if affine_matrix is None:
        return None, None
    aligned = cv2.warpAffine(
        frame, affine_matrix, (output_size, output_size),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132),
    )
    return aligned, affine_matrix


def _paste_back(
    frame: np.ndarray,
    enhanced: np.ndarray,
    affine_matrix: np.ndarray,
    output_size: int,
    blend_weight: float = 1.0,
) -> np.ndarray:
    h, w = frame.shape[:2]
    inv = cv2.invertAffineTransform(affine_matrix)
    inv_restored = cv2.warpAffine(
        enhanced, inv, (w, h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )

    # Feathered mask (5 % border softening)
    mask = np.ones((output_size, output_size), dtype=np.float32)
    border = max(1, int(output_size * 0.05))
    ramp = np.linspace(0.0, 1.0, border, dtype=np.float32)
    mask[:border, :]  *= ramp[:, None]
    mask[-border:, :] *= ramp[::-1, None]
    mask[:, :border]  *= ramp[None, :]
    mask[:, -border:] *= ramp[None, ::-1]

    if blend_weight < 1.0:
        mask *= blend_weight

    mask_3c = np.stack([mask] * 3, axis=-1)
    inv_mask = cv2.warpAffine(
        mask_3c, inv, (w, h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    inv_mask = np.clip(inv_mask, 0.0, 1.0)

    result = frame.astype(np.float32) * (1.0 - inv_mask) + inv_restored.astype(np.float32) * inv_mask
    return np.clip(result, 0, 255).astype(np.uint8)


def _preprocess(face_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb / 255.0 - 0.5) / 0.5
    return np.expand_dims(np.transpose(rgb, (2, 0, 1)), 0)


def _postprocess(output: np.ndarray) -> np.ndarray:
    face = np.squeeze(output)
    face = np.transpose(face, (1, 2, 0))
    face = np.clip((face + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(face, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Main single-face enhancement helper
# ---------------------------------------------------------------------------

def enhance_face_onnx(
    frame: np.ndarray,
    face: object,
    session: onnxruntime.InferenceSession,
    input_size: int,
) -> np.ndarray:
    """Align *face* from *frame*, enhance via *session*, paste back.

    Parameters
    ----------
    frame:      BGR uint8 source frame (not modified in-place).
    face:       InsightFace face object (must have ``.kps`` keypoints).
    session:    Pre-loaded ONNX Runtime inference session.
    input_size: Model's expected square input side-length (256 or 512).
    """
    if not hasattr(face, "kps") or face.kps is None:
        return frame
    landmarks_5 = face.kps.astype(np.float32)
    if landmarks_5.shape[0] < 5:
        return frame

    aligned, affine = _align_face(frame, landmarks_5, output_size=input_size)
    if aligned is None:
        return frame

    inp_name = session.get_inputs()[0].name
    tensor = _preprocess(aligned)
    output = session.run(None, {inp_name: tensor})[0]
    enhanced = _postprocess(output)

    # Resize back to input_size if model up-scales internally
    eh, ew = enhanced.shape[:2]
    if eh != input_size or ew != input_size:
        enhanced = cv2.resize(enhanced, (input_size, input_size), interpolation=cv2.INTER_LANCZOS4)

    enhance_weight = getattr(config, "ENHANCE_WEIGHT", 0.6)
    return _paste_back(frame.copy(), enhanced, affine, input_size, blend_weight=1.0 - enhance_weight)
