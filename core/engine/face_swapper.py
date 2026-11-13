import os
import cv2
import numpy as np
import insightface
import threading
from typing import Any, List
from core.config import SWAPPER_MODEL
import core.config as config
from core.engine.gpu_processing import gpu_add_weighted, gpu_resize, gpu_sharpen
from core.engine.face_masking import (
    create_face_mask,
    create_lower_mouth_mask,
    apply_mask_area,
    create_eyes_mask,
    create_eyebrows_mask,
)

swapper_ = None
THREAD_LOCK = threading.Lock()
PREVIOUS_FRAME_RESULT = None


def get_face_swapper():
    global swapper_
    with THREAD_LOCK:
        if swapper_ is None:
            providers = getattr(
                config,
                "EXECUTION_PROVIDERS",
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            # Keep the configured standard model on both CPU and CUDA.
            # The bundled FP16 conversion fails source-identity checks; merely
            # detecting CUDA is not sufficient to select a different model.
            model_path = str(config.SWAPPER_MODEL)
            swapper_ = insightface.model_zoo.get_model(model_path, providers=providers)
    return swapper_


def swap_face(source_face, target_face, frame):
    if source_face is None or target_face is None:
        return frame
    face_swapper = get_face_swapper()
    if face_swapper is None:
        return frame

    opacity = getattr(config, "OPACITY", 1.0)
    mouth_mask_enabled = getattr(config, "MOUTH_MASK_ENABLED", False)
    eyes_mask_enabled = getattr(config, "EYES_MASK_ENABLED", False)
    eyebrows_mask_enabled = getattr(config, "EYEBROWS_MASK_ENABLED", False)

    original_frame = (
        frame.copy()
        if (
            opacity < 1.0
            or mouth_mask_enabled
            or eyes_mask_enabled
            or eyebrows_mask_enabled
        )
        else frame
    )

    temp_frame = frame
    if temp_frame.dtype != np.uint8:
        temp_frame = np.clip(temp_frame, 0, 255).astype(np.uint8)

    if not temp_frame.flags["C_CONTIGUOUS"]:
        temp_frame = np.ascontiguousarray(temp_frame)

    try:
        swapped_frame_raw = face_swapper.get(
            temp_frame, target_face, source_face, paste_back=True
        )
        if swapped_frame_raw is None or not isinstance(swapped_frame_raw, np.ndarray):
            return original_frame
        if swapped_frame_raw.shape != temp_frame.shape:
            swapped_frame_raw = gpu_resize(
                swapped_frame_raw, (temp_frame.shape[1], temp_frame.shape[0])
            )
        swapped_frame = np.clip(swapped_frame_raw, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"Error during swap: {e}")
        return original_frame

    # Landmarks are computed only when a mask is enabled.
    if (mouth_mask_enabled or eyes_mask_enabled or eyebrows_mask_enabled
            or getattr(config, "POISSON_BLEND_ENABLED", False)):
        from core.face_analyser import ensure_face_landmarks
        ensure_face_landmarks(original_frame, target_face)

    # Masking
    face_mask = None
    if (
        mouth_mask_enabled
        or eyes_mask_enabled
        or eyebrows_mask_enabled
        or getattr(config, "POISSON_BLEND_ENABLED", False)
    ):
        face_mask = create_face_mask(target_face, original_frame)

    # Poisson Blending
    if getattr(config, "POISSON_BLEND_ENABLED", False):
        if face_mask is not None:
            y_indices, x_indices = np.where(face_mask > 0)
            if len(x_indices) > 0 and len(y_indices) > 0:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
                src_crop = swapped_frame[y_min : y_max + 1, x_min : x_max + 1]
                mask_crop = face_mask[y_min : y_max + 1, x_min : x_max + 1]
                try:
                    swapped_frame = cv2.seamlessClone(
                        src_crop, original_frame, mask_crop, center, cv2.NORMAL_CLONE
                    )
                except Exception as e:
                    print(f"Poisson blending failed: {e}")


    if mouth_mask_enabled:
        mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon = (
            create_lower_mouth_mask(target_face, original_frame)
        )
        if mouth_cutout is not None and mouth_box != (0, 0, 0, 0):
            swapped_frame = apply_mask_area(
                swapped_frame, mouth_cutout, mouth_box, face_mask, lower_lip_polygon, region_mask=mouth_mask
            )

    if eyes_mask_enabled:
        mask, cutout, box, polygon = create_eyes_mask(target_face, original_frame)
        if cutout is not None and box != (0, 0, 0, 0):
            swapped_frame = apply_mask_area(
                swapped_frame, cutout, box, face_mask, polygon, region_mask=mask
            )

    if eyebrows_mask_enabled:
        mask, cutout, box, polygon = create_eyebrows_mask(target_face, original_frame)
        if cutout is not None and box != (0, 0, 0, 0):
            swapped_frame = apply_mask_area(
                swapped_frame, cutout, box, face_mask, polygon, region_mask=mask
            )

    if opacity >= 1.0:
        return swapped_frame.astype(np.uint8)

    final_swapped_frame = gpu_add_weighted(
        original_frame.astype(np.uint8),
        1 - opacity,
        swapped_frame.astype(np.uint8),
        opacity,
        0,
    )
    return final_swapped_frame.astype(np.uint8)


def apply_post_processing(current_frame, swapped_face_bboxes):
    global PREVIOUS_FRAME_RESULT
    processed_frame = current_frame.copy()
    sharpness_value = getattr(config, "SHARPNESS", 0.0)
    if sharpness_value > 0.0 and swapped_face_bboxes:
        height, width = processed_frame.shape[:2]
        for bbox in swapped_face_bboxes:
            if not hasattr(bbox, "__iter__") or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            try:
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(width, int(x2)), min(height, int(y2))
            except ValueError:
                continue

            if x2 <= x1 or y2 <= y1:
                continue

            face_region = processed_frame[y1:y2, x1:x2]
            if face_region.size == 0:
                continue

            try:
                sharpened_region = gpu_sharpen(
                    face_region, strength=sharpness_value, sigma=3
                )
                processed_frame[y1:y2, x1:x2] = sharpened_region
            except cv2.error:
                pass

    enable_interpolation = getattr(config, "ENABLE_INTERPOLATION", False)
    interpolation_weight = getattr(config, "INTERPOLATION_WEIGHT", 0.2)
    final_frame = processed_frame

    if enable_interpolation and 0 < interpolation_weight < 1:
        if (
            PREVIOUS_FRAME_RESULT is not None
            and PREVIOUS_FRAME_RESULT.shape == processed_frame.shape
            and PREVIOUS_FRAME_RESULT.dtype == processed_frame.dtype
        ):
            try:
                final_frame = gpu_add_weighted(
                    PREVIOUS_FRAME_RESULT,
                    interpolation_weight,
                    processed_frame,
                    1.0 - interpolation_weight,
                    0,
                )
                final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)
            except cv2.error:
                final_frame = processed_frame
                PREVIOUS_FRAME_RESULT = None
            PREVIOUS_FRAME_RESULT = processed_frame.copy()
        else:
            PREVIOUS_FRAME_RESULT = processed_frame.copy()
    else:
        PREVIOUS_FRAME_RESULT = None

    return final_frame


def detect_and_swap(source_face, frame, face_analyser, target_embeddings=None):
    if getattr(config, "OPACITY", 1.0) == 0:
        global PREVIOUS_FRAME_RESULT
        PREVIOUS_FRAME_RESULT = None
        return frame, 0

    if face_analyser is None:
        from core.face_analyser import get_face_analyser
        face_analyser = get_face_analyser(live_mode=True)
    faces = face_analyser.get(frame)
    if not faces:
        PREVIOUS_FRAME_RESULT = None
        return frame, 0

    face_count = len(faces)
    processed_frame = frame
    swapped_face_bboxes = []
    current_swap_target = processed_frame.copy()

    many_faces = getattr(config, "MANY_FACES", False)
    map_faces = getattr(config, "MAP_FACES", False)
    simple_map = getattr(config, "simple_map", None)

    MATCH_THRESHOLD = 0.45

    def _matches_selection(face):
        if target_embeddings is None or len(target_embeddings) == 0:
            return True
        if face.normed_embedding is None:
            return False
        sims = [float(np.dot(face.normed_embedding, ref)) for ref in target_embeddings]
        return max(sims) >= MATCH_THRESHOLD

    source_target_pairs = []

    if many_faces:
        if source_face:
            for target_face in faces:
                if _matches_selection(target_face):
                    source_target_pairs.append((source_face, target_face))
    elif map_faces and simple_map:
        source_faces = simple_map.get("source_faces", [])
        target_embeddings_map = simple_map.get("target_embeddings", [])
        if (
            source_faces
            and target_embeddings_map
            and len(source_faces) == len(target_embeddings_map)
        ):
            from core.cluster_analysis import find_closest_centroid

            if len(faces) <= len(target_embeddings_map):
                for detected_face in faces:
                    if detected_face.normed_embedding is None:
                        continue
                    if not _matches_selection(detected_face):
                        continue
                    closest_idx, _ = find_closest_centroid(
                        target_embeddings_map, detected_face.normed_embedding
                    )
                    if 0 <= closest_idx < len(source_faces):
                        source_target_pairs.append(
                            (source_faces[closest_idx], detected_face)
                        )
            else:
                detected_embeddings = [
                    f.normed_embedding for f in faces if f.normed_embedding is not None
                ]
                detected_faces_with_embedding = [
                    f for f in faces if f.normed_embedding is not None
                ]
                if detected_embeddings:
                    for i, target_embedding in enumerate(target_embeddings_map):
                        if 0 <= i < len(source_faces):
                            closest_idx, _ = find_closest_centroid(
                                detected_embeddings, target_embedding
                            )
                            if 0 <= closest_idx < len(detected_faces_with_embedding):
                                source_target_pairs.append(
                                    (
                                        source_faces[i],
                                        detected_faces_with_embedding[closest_idx],
                                    )
                                )
    else:
        candidates = [f for f in faces if _matches_selection(f)]
        target_face = min(candidates, key=lambda x: x.bbox[0]) if candidates else None
        if source_face and target_face:
            source_target_pairs.append((source_face, target_face))

    for src_face, tgt_face in source_target_pairs:
        current_swap_target = swap_face(src_face, tgt_face, current_swap_target)
        if (
            tgt_face is not None
            and hasattr(tgt_face, "bbox")
            and tgt_face.bbox is not None
        ):
            swapped_face_bboxes.append(tgt_face.bbox.astype(int))

    processed_frame = current_swap_target
    final_frame = apply_post_processing(processed_frame, swapped_face_bboxes)
    return final_frame, face_count
