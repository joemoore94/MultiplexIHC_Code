"""Tests for registration.features module."""
import numpy as np
import pytest


def test_create_detector_sift():
    """SIFT detector should be available in mainline OpenCV."""
    pytest.importorskip('cv2')
    from lib.features import create_detector
    det = create_detector("SIFT")
    assert det is not None


def test_create_detector_invalid():
    from lib.features import create_detector
    with pytest.raises(ValueError):
        create_detector("INVALID")


def test_detect_features_on_noise():
    """Feature detection should find keypoints in a textured image."""
    pytest.importorskip('cv2')
    from lib.features import detect_features

    rng = np.random.RandomState(42)
    img = rng.randint(0, 256, (512, 512), dtype=np.uint8)
    kp, desc = detect_features(img, max_features=100)
    assert len(kp) > 0
    assert desc is not None


def test_detect_features_blank_image():
    """A blank image should produce few or no keypoints."""
    pytest.importorskip('cv2')
    from lib.features import detect_features

    img = np.zeros((512, 512), dtype=np.uint8)
    kp, desc = detect_features(img)
    # Blank image: no features expected
    assert len(kp) == 0


def test_match_features_identity():
    """Matching an image's features against itself should yield many matches."""
    pytest.importorskip('cv2')
    from lib.features import detect_features, match_features

    rng = np.random.RandomState(42)
    img = rng.randint(0, 256, (512, 512), dtype=np.uint8)
    kp, desc = detect_features(img, max_features=500)

    if desc is None:
        pytest.skip('No features detected')

    assert desc is not None  # help mypy narrow the type
    matches = match_features(desc, desc, ratio=0.8)
    # Self-matching should be perfect (or near-perfect)
    assert len(matches) > len(kp) * 0.5


def test_register_single_marker_no_features():
    """Blank moving image should return no_features status."""
    pytest.importorskip('cv2')
    from lib.features import register_single_marker

    ref = np.random.RandomState(42).randint(0, 256, (256, 256, 3), dtype=np.uint8)
    obj = np.zeros((256, 256, 3), dtype=np.uint8)

    M, status = register_single_marker(ref, obj)
    assert status == "no_features"
    assert M is None
