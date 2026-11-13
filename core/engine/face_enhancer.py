import cv2
import threading
import numpy as np
from pathlib import Path

import onnxruntime
import core.config as config

_ENHANCER = None
_LOCK = threading.Lock()

FFHQ_TEMPLATE_512 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19366],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)

def get_face_enhancer() -> onnxruntime.InferenceSession:
    global _ENHANCER
    with _LOCK:
        if _ENHANCER is None:
            # Auto-select the higher-quality gfpgan-1024 if present, fallback to v1.4
            import os
            model_path = str(config.GFPGAN_1024_MODEL)
            if not os.path.exists(model_path):
                model_path = str(config.ENHANCER_MODEL)
                
            if not Path(model_path).exists():
                raise FileNotFoundError(f"GFPGAN ONNX Model not found. Checked {config.GFPGAN_1024_MODEL} and {config.ENHANCER_MODEL}")
                raise FileNotFoundError(f"ONNX Model not found at {model_path}")
            
            providers = getattr(config, "EXECUTION_PROVIDERS", ['CUDAExecutionProvider', 'CPUExecutionProvider'])
            session_options = onnxruntime.SessionOptions()
            session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            _ENHANCER = onnxruntime.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers,
            )
    return _ENHANCER

def _align_face(frame: np.ndarray, landmarks_5: np.ndarray, output_size: int) -> tuple:
    scale = output_size / 512.0
    template = FFHQ_TEMPLATE_512 * scale
    affine_matrix, _ = cv2.estimateAffinePartial2D(landmarks_5, template, method=cv2.LMEDS)
    if affine_matrix is None:
        return None, None
    aligned_face = cv2.warpAffine(
        frame, affine_matrix, (output_size, output_size),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132)
    )
    return aligned_face, affine_matrix

def _paste_back(frame: np.ndarray, enhanced_face: np.ndarray, affine_matrix: np.ndarray, output_size: int, weight: float) -> np.ndarray:
    h, w = frame.shape[:2]
    inv_matrix = cv2.invertAffineTransform(affine_matrix)
    inv_restored = cv2.warpAffine(
        enhanced_face, inv_matrix, (w, h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    
    face_mask = np.ones((output_size, output_size), dtype=np.float32)
    border = max(1, int(output_size * 0.05))
    ramp_up = np.linspace(0.0, 1.0, border, dtype=np.float32)
    ramp_down = np.linspace(1.0, 0.0, border, dtype=np.float32)

    face_mask[:border, :] *= ramp_up[:, None]
    face_mask[-border:, :] *= ramp_down[:, None]
    face_mask[:, :border] *= ramp_up[None, :]
    face_mask[:, -border:] *= ramp_down[None, :]

    if weight < 1.0:
        face_mask *= weight

    face_mask_3c = np.stack([face_mask] * 3, axis=-1)
    inv_mask = cv2.warpAffine(
        face_mask_3c, inv_matrix, (w, h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    inv_mask = np.clip(inv_mask, 0.0, 1.0)

    result = (frame.astype(np.float32) * (1.0 - inv_mask) + inv_restored.astype(np.float32) * inv_mask)
    return np.clip(result, 0, 255).astype(np.uint8)

def _preprocess_face(aligned_face: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = rgb / 255.0
    rgb = (rgb - 0.5) / 0.5
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw, axis=0)

def _postprocess_face(output: np.ndarray) -> np.ndarray:
    face = np.squeeze(output)
    face = np.transpose(face, (1, 2, 0))
    face = (face + 1.0) / 2.0
    face = np.clip(face * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

def enhance_faces(frame: np.ndarray, faces: list, frame_size: tuple) -> np.ndarray:
    session = get_face_enhancer()
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    try:
        align_size = int(input_info.shape[2])
        if align_size <= 0: align_size = 512
    except (ValueError, TypeError, IndexError):
        align_size = 512

    result_frame = frame.copy()
    
    for face in faces:
        if not hasattr(face, "kps") or face.kps is None:
            continue
            
        landmarks_5 = face.kps.astype(np.float32)
        if landmarks_5.shape[0] < 5:
            continue
            
        aligned_face, affine_matrix = _align_face(result_frame, landmarks_5, output_size=align_size)
        if aligned_face is None or affine_matrix is None:
            continue
            
        try:
            input_tensor = _preprocess_face(aligned_face)
            output_tensor = session.run(None, {input_name: input_tensor})[0]
            enhanced_bgr = _postprocess_face(output_tensor)
            
            eh, ew = enhanced_bgr.shape[:2]
            if eh != align_size or ew != align_size:
                enhanced_bgr = cv2.resize(enhanced_bgr, (align_size, align_size), interpolation=cv2.INTER_LANCZOS4)
                
            blend_weight = 1.0 - float(np.clip(config.ENHANCE_WEIGHT, 0.0, 1.0))
            result_frame = _paste_back(result_frame, enhanced_bgr, affine_matrix, output_size=align_size, weight=blend_weight)
        except Exception as e:
            print(f"Error enhancing face: {e}")
            continue
            
    return result_frame
