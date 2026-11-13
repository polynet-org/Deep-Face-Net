import cv2
import numpy as np
from typing import Any
import core.config as config

Face = Any
Frame = np.ndarray

def apply_color_transfer(source, target):
    """
    Apply color transfer from target to source image using LAB color space.
    Uses float32 throughout for performance (sufficient precision for 8-bit images).
    """
    # Convert to float32 [0,1] range for proper LAB conversion
    source_f32 = source.astype(np.float32) / 255.0
    target_f32 = target.astype(np.float32) / 255.0

    source_lab = cv2.cvtColor(source_f32, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_f32, cv2.COLOR_BGR2LAB)

    source_mean, source_std = cv2.meanStdDev(source_lab)
    target_mean, target_std = cv2.meanStdDev(target_lab)

    source_mean = source_mean.reshape(1, 1, 3).astype(np.float32)
    source_std = np.maximum(source_std.reshape(1, 1, 3), 1e-6).astype(np.float32)
    target_mean = target_mean.reshape(1, 1, 3).astype(np.float32)
    target_std = target_std.reshape(1, 1, 3).astype(np.float32)

    result_lab = (source_lab - source_mean) * (target_std / source_std) + target_mean
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    return np.clip(result_bgr * 255.0, 0, 255).astype(np.uint8)

def _landmarks(face):
    points = getattr(face, "landmark_2d_106", None)
    if points is None:
        return None
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (106, 2) or not np.isfinite(points).all():
        return None
    return points


def create_face_mask(face: Face, frame: Frame) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    points = _landmarks(face)
    if points is not None:
        hull = cv2.convexHull(points[:33].astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        mask = cv2.GaussianBlur(mask, (5, 5), 1)
    return mask


def _feature_mask(face, frame, groups, expansion=1.0):
    """Rasterize separate convex features in a clipped, feathered ROI.

    InsightFace's landmark indices are not contour order; filling them as
    one polygon creates crossing edges (particularly around the mouth).
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    empty = (mask, None, (0, 0, 0, 0), None)
    points = _landmarks(face)
    if points is None or frame.size == 0:
        return empty
    expansion = float(np.clip(expansion, 0.1, 4.0))
    polygons = []
    for indices in groups:
        feature = points[indices]
        center = feature.mean(axis=0)
        hull = cv2.convexHull((center + (feature - center) * expansion).astype(np.int32))
        if cv2.contourArea(hull) > 0:
            polygons.append(hull.reshape(-1, 2))
    if not polygons:
        return empty
    all_points = np.concatenate(polygons)
    extent = np.ptp(all_points, axis=0)
    ratio = max(1.0, float(getattr(config, "MASK_FEATHER_RATIO", 10)))
    radius = max(1, min(15, int(min(extent) / ratio)))
    padding = radius * 2 + 1
    h, w = frame.shape[:2]
    lo = np.maximum(all_points.min(axis=0) - padding, [0, 0])
    hi = np.minimum(all_points.max(axis=0) + padding + 1, [w, h])
    x1, y1 = map(int, lo)
    x2, y2 = map(int, hi)
    if x2 <= x1 or y2 <= y1:
        return empty
    roi = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    for polygon in polygons:
        cv2.fillConvexPoly(roi, polygon - [x1, y1], 255)
    roi = cv2.GaussianBlur(roi, (2*radius+1, 2*radius+1), 0)
    mask[y1:y2, x1:x2] = roi
    return mask, frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2), all_points


def create_lower_mouth_mask(face: Face, frame: Frame):
    return _feature_mask(face, frame, [slice(52, 72)],
                         1 + config.MOUTH_MASK_SIZE / 100.0 * 2.5)


def create_eyes_mask(face: Face, frame: Frame):
    return _feature_mask(face, frame, [slice(87, 97), slice(33, 43)],
                         1 + config.MASK_DOWN_SIZE * config.EYES_MASK_SIZE)


def create_eyebrows_mask(face: Face, frame: Frame):
    return _feature_mask(face, frame, [slice(97, 106), slice(43, 52)],
                         1 + config.EYEBROWS_MASK_SIZE)


def apply_mask_area(frame, cutout, box, face_mask, polygon, region_mask=None):
    """Restore original feature pixels using the already feathered mask.

    Avoid repeated LAB transfer (which discolors teeth/eyes), polygon
    reconstruction, and double alpha blending of the same region.
    """
    if cutout is None or face_mask is None or polygon is None:
        return frame
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        return frame
    roi = frame[y1:y2, x1:x2]
    if cutout.shape != roi.shape:
        return frame
    mask = region_mask if region_mask is not None else face_mask
    alpha = mask[y1:y2, x1:x2].astype(np.float32)[..., None] / 255.0
    frame[y1:y2, x1:x2] = np.clip(
        cutout * alpha + roi * (1.0 - alpha), 0, 255
    ).astype(np.uint8)
    return frame
