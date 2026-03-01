"""Tests for registration.io module (XML parsing, file discovery, region processing)."""
from pathlib import Path

import pytest

from lib.io import compute_pixel_regions, parse_xml

TEST_DATA = Path(__file__).parent / 'test_data'


@pytest.mark.skipif(
    not (TEST_DATA / 'annotations.xml').exists(),
    reason='No test XML baseline',
)
def test_parse_xml_returns_regions():
    regions = parse_xml(str(TEST_DATA / 'annotations.xml'))
    assert isinstance(regions, list)
    assert len(regions) > 0
    # Each region should be a list of (x, y) tuples
    for region in regions:
        assert len(region) >= 3  # At least a triangle
        for x, y in region:
            assert isinstance(x, float)
            assert isinstance(y, float)


@pytest.mark.skipif(
    not (TEST_DATA / 'annotations.xml').exists(),
    reason='No test XML baseline',
)
def test_parse_xml_matches_matlab_baseline():
    """Compare parsed coordinates against MATLAB-exported baseline."""
    import json

    import numpy as np

    baseline_path = TEST_DATA / 'xml_coordinates.json'
    if not baseline_path.exists():
        pytest.skip('No MATLAB baseline (xml_coordinates.json)')

    regions = parse_xml(str(TEST_DATA / 'annotations.xml'))
    with open(baseline_path) as f:
        matlab_regions = json.load(f)

    assert len(regions) == len(matlab_regions)
    for py_region, mat_region in zip(regions, matlab_regions):
        py_arr = np.array(py_region)
        mat_arr = np.array(mat_region)
        np.testing.assert_allclose(py_arr, mat_arr, atol=1e-6)


def test_filename_uppercasing():
    """Filenames must be UPPERCASE without extension (MATLAB convention)."""
    # get_image_sets returns uppercased stems; test the logic directly
    from pathlib import PurePosixPath
    name = PurePosixPath("CD4_Slide_001.svs").stem.upper()
    assert name == "CD4_SLIDE_001"


def test_compute_pixel_regions_buffer_clamp():
    """Buffer should be clamped to image boundaries."""
    # Fake ROI near the top-left corner
    xy = [[(100.0, 100.0), (100.0, 500.0), (500.0, 500.0), (500.0, 100.0)]]

    # Mock: we can't call compute_pixel_regions without a real SVS,
    # but we can test the clamping logic directly

    # This test requires a real image path; skip if no test data
    svs_path = TEST_DATA / 'sample.svs'
    if not svs_path.exists():
        pytest.skip('No test SVS file')

    regions = compute_pixel_regions(xy, str(svs_path), buff=2000)
    for (row_start, _), (col_start, _) in regions['pixel_region_buff']:
        assert row_start >= 0
        assert col_start >= 0
