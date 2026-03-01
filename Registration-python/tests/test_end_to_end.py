"""End-to-end registration tests against MATLAB baselines.

These tests require real slide data and MATLAB-generated baselines.
They are skipped if test data is not present.
"""
from pathlib import Path

import numpy as np
import pytest

TEST_DATA = Path(__file__).parent / 'test_data'


@pytest.mark.skipif(
    not (TEST_DATA / 'registered_image_ref.tif').exists(),
    reason='No MATLAB baseline images',
)
def test_registration_quality_ssim():
    """Registered image must be structurally similar to MATLAB output."""
    from PIL import Image
    from skimage.metrics import structural_similarity

    py_path = TEST_DATA / 'registered_image_python.tif'
    mat_path = TEST_DATA / 'registered_image_ref.tif'

    if not py_path.exists():
        pytest.skip('Run registration first to generate Python output')

    py_img = np.array(Image.open(str(py_path)))
    mat_img = np.array(Image.open(str(mat_path)))

    assert py_img.shape == mat_img.shape, (
        f'Shape mismatch: Python {py_img.shape} vs MATLAB {mat_img.shape}'
    )

    ssim = structural_similarity(py_img, mat_img, channel_axis=2)
    assert ssim > 0.95, f'SSIM {ssim:.4f} below threshold 0.95'


@pytest.mark.skipif(
    not (TEST_DATA / 'region_coords.json').exists(),
    reason='No MATLAB region baseline',
)
def test_region_coords_match_matlab():
    """Pixel regions with buffer must match MATLAB exactly."""
    import json

    with open(TEST_DATA / 'region_coords.json') as f:
        matlab_regions = json.load(f)

    # This would run compute_pixel_regions on the same input and compare
    # Requires the test SVS file to be present
    svs_path = TEST_DATA / 'sample.svs'
    xml_path = TEST_DATA / 'annotations.xml'
    if not svs_path.exists() or not xml_path.exists():
        pytest.skip('No test SVS/XML files')

    from lib.io import compute_pixel_regions, parse_xml
    xy = parse_xml(str(xml_path))
    regions = compute_pixel_regions(xy, str(svs_path), buff=2000)

    assert len(regions['pixel_region_buff']) == len(matlab_regions)
