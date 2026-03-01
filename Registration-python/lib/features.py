"""Feature detection, matching, and transform estimation.

Supports SIFT (default, patent-free) and SURF (optional, requires opencv-contrib-python).
Implements the dual-channel fallback logic from register_SURF.m.
"""
from typing import Optional

import cv2
import numpy as np

import config


def create_detector(method: Optional[str] = None):
    """Create a feature detector instance.

    Args:
        method: "SIFT" or "SURF".  Defaults to config.FEATURE_DETECTOR.

    Returns:
        OpenCV feature detector.
    """
    method = method or config.FEATURE_DETECTOR

    if method == "SIFT":
        return cv2.SIFT_create(nfeatures=config.MAX_FEATURES)
    elif method == "SURF":
        try:
            return cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        except AttributeError as err:
            raise RuntimeError(
                "SURF requires opencv-contrib-python with nonfree modules. "
                "Install it or switch to SIFT (recommended)."
            ) from err
    else:
        raise ValueError(f"Unknown feature detector: {method!r}")


def detect_features(
    gray_image: np.ndarray,
    detector=None,
    max_features: Optional[int] = None,
) -> tuple[list, Optional[np.ndarray]]:
    """Detect keypoints and compute descriptors.

    Args:
        gray_image: 2-D uint8 array.
        detector: Pre-created detector (or None to create from config).
        max_features: Retain only the N strongest features.

    Returns:
        (keypoints, descriptors).  descriptors is None if nothing found.
    """
    if detector is None:
        detector = create_detector()
    max_features = max_features or config.MAX_FEATURES

    keypoints, descriptors = detector.detectAndCompute(gray_image, None)

    # Retain strongest N features (mirrors MATLAB selectStrongest)
    if keypoints and len(keypoints) > max_features:
        keypoints, descriptors = _select_strongest(keypoints, descriptors, max_features)

    return keypoints, descriptors


def _select_strongest(keypoints, descriptors, n):
    """Keep the n keypoints with the largest response."""
    indices = sorted(
        range(len(keypoints)),
        key=lambda i: keypoints[i].response,
        reverse=True,
    )[:n]
    keypoints = [keypoints[i] for i in indices]
    descriptors = descriptors[indices]
    return keypoints, descriptors


def match_features(
    desc_ref: Optional[np.ndarray],
    desc_obj: Optional[np.ndarray],
    ratio: Optional[float] = None,
) -> list:
    """Match descriptors using BFMatcher with Lowe's ratio test.

    Args:
        desc_ref: Descriptors from reference image (or None if no features).
        desc_obj: Descriptors from moving image (or None if no features).
        ratio: Ratio test threshold.  Defaults to config.MATCH_RATIO.

    Returns:
        List of good DMatch objects.
    """
    ratio = ratio or config.MATCH_RATIO

    if desc_ref is None or desc_obj is None:
        return []

    # SIFT uses float32 -> L2; ORB uses uint8 -> HAMMING
    norm = cv2.NORM_L2 if desc_ref.dtype != np.uint8 else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm, crossCheck=False)

    matches = bf.knnMatch(desc_ref, desc_obj, k=2)
    good = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def estimate_transform(
    kp_ref: list,
    kp_obj: list,
    matches: list,
    max_trials: Optional[int] = None,
    confidence: Optional[float] = None,
    max_distance: Optional[float] = None,
) -> tuple[Optional[np.ndarray], int]:
    """Estimate a similarity transform from matched keypoints using RANSAC.

    Uses cv2.estimateAffinePartial2D which estimates a 4-DOF similarity
    transform (rotation, uniform scale, translation) -- equivalent to MATLAB's
    estimateGeometricTransform(..., 'similarity').

    Args:
        kp_ref: Reference keypoints.
        kp_obj: Moving image keypoints.
        matches: DMatch list from match_features().
        max_trials: RANSAC iterations.
        confidence: RANSAC confidence.
        max_distance: Inlier distance threshold.

    Returns:
        (M, num_inliers) where M is a 2x3 affine matrix or None on failure.
    """
    max_trials = max_trials or config.RANSAC_MAX_TRIALS_PRIMARY
    confidence = confidence or config.RANSAC_CONFIDENCE
    max_distance = max_distance or config.RANSAC_MAX_DISTANCE

    if not matches:
        return None, 0

    pts_ref = np.array([kp_ref[m.queryIdx].pt for m in matches], dtype=np.float32)
    pts_obj = np.array([kp_obj[m.trainIdx].pt for m in matches], dtype=np.float32)

    if len(pts_ref) < 3:
        return None, 0

    M, inliers = cv2.estimateAffinePartial2D(
        pts_obj, pts_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=max_distance,
        maxIters=max_trials,
        confidence=confidence,
    )

    num_inliers = int(inliers.sum()) if inliers is not None else 0
    return M, num_inliers


def register_single_marker(
    ref_rgb: np.ndarray,
    obj_rgb: np.ndarray,
    detector=None,
) -> tuple[Optional[np.ndarray], str]:
    """Register a single marker image against the nuclei reference.

    Implements the dual-channel fallback: tries blue channel first, falls back
    to red channel if <= FALLBACK_KP_THRESHOLD inlier keypoints, and picks
    whichever channel gave more inliers.

    Args:
        ref_rgb: Nuclei reference image (H, W, 3), uint8.
        obj_rgb: Moving marker image (H, W, 3), uint8.
        detector: Pre-created feature detector (or None).

    Returns:
        (transform_matrix, status) where:
        - transform_matrix: 2x3 affine matrix, or None on failure
        - status: "success", "no_features", or "failed"
    """
    if detector is None:
        detector = create_detector()

    # Channel indices: MATLAB blue=(:,:,3), red=(:,:,2) -> Python blue=2, red=1
    ref_blue = ref_rgb[:, :, 2]
    ref_red = ref_rgb[:, :, 1]
    obj_blue = obj_rgb[:, :, 2]

    # Detect features in moving image (blue channel)
    kp_obj, desc_obj = detect_features(obj_blue, detector)
    if not kp_obj:
        return None, "no_features"

    # --- Primary: blue channel reference ---
    kp_ref_blue, desc_ref_blue = detect_features(ref_blue, detector)
    matches_blue = match_features(desc_ref_blue, desc_obj)
    M_blue, kp_blue = estimate_transform(
        kp_ref_blue, kp_obj, matches_blue,
        max_trials=config.RANSAC_MAX_TRIALS_PRIMARY,
    )

    best_M = M_blue
    best_kp = kp_blue

    # --- Fallback: red channel reference ---
    if best_kp <= config.FALLBACK_KP_THRESHOLD:
        kp_ref_red, desc_ref_red = detect_features(ref_red, detector)
        matches_red = match_features(desc_ref_red, desc_obj)
        M_red, kp_red = estimate_transform(
            kp_ref_red, kp_obj, matches_red,
            max_trials=config.RANSAC_MAX_TRIALS_FALLBACK,
        )

        if kp_red > best_kp:
            best_M = M_red
            best_kp = kp_red

    if best_M is None:
        return None, "failed"

    # Check for degenerate (nearly singular) transform
    # Mirrors MATLAB's nearlySingularMatrix warning detection
    det = best_M[0, 0] * best_M[1, 1] - best_M[0, 1] * best_M[1, 0]
    if abs(det) < 1e-6:
        return None, "failed"

    return best_M, "success"
