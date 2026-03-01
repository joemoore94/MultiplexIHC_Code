"""I/O utilities: XML parsing, SVS/image discovery, region processing, TIFF writing.

Consolidates the functionality of parse_xml.m, get_image_sets.m, and sort_regions.m.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import openslide
except ImportError:
    openslide = None

try:
    import tifffile
except ImportError:
    tifffile = None


# ---------------------------------------------------------------------------
# XML Parsing (parse_xml.m)
# ---------------------------------------------------------------------------

def parse_xml(xml_path: str) -> list[list[tuple[float, float]]]:
    """Parse Aperio/Leica XML annotation and return ROI vertex coordinates.

    Args:
        xml_path: Path to the XML file.

    Returns:
        List of regions.  Each region is a list of (x, y) float tuples.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    regions: list[list[tuple[float, float]]] = []
    for region in root.findall('.//Region'):
        verts = []
        for vertex in region.findall('.//Vertex'):
            x_attr = vertex.get('X')
            y_attr = vertex.get('Y')
            if x_attr is None or y_attr is None:
                continue
            x = float(x_attr)
            y = float(y_attr)
            verts.append((x, y))
        if verts:
            regions.append(verts)
    return regions


# ---------------------------------------------------------------------------
# Image Set Discovery (get_image_sets.m)
# ---------------------------------------------------------------------------

def get_image_sets(slide_dir: str) -> tuple[list[str], Optional[str], list[str], int]:
    """Discover SVS files and XML annotation for a slide directory.

    Filenames are stored UPPERCASE without extension to match MATLAB convention.

    Args:
        slide_dir: Directory containing .svs and .xml files.

    Returns:
        (svs_paths, xml_path, filenames, nuc_index)
        - svs_paths: sorted list of full SVS file paths
        - xml_path: path to the XML annotation file (or None)
        - filenames: UPPERCASE names without extension (e.g. "CD4_SLIDE_001")
        - nuc_index: index into svs_paths/filenames for the nuclei reference
    """
    p = Path(slide_dir)

    svs_files = sorted(p.glob('*.svs'))
    svs_paths = [str(f) for f in svs_files]
    filenames = [f.stem.upper() for f in svs_files]

    xml_files = list(p.glob('*.xml'))
    xml_path = str(xml_files[0]) if xml_files else None

    # Identify nuclei reference: the SVS whose name contains the XML stem
    nuc_index = 0
    if xml_path is not None:
        xml_stem = Path(xml_path).stem.upper()
        for i, name in enumerate(filenames):
            if xml_stem in name:
                nuc_index = i
                break

    return svs_paths, xml_path, filenames, nuc_index


# ---------------------------------------------------------------------------
# Region Processing (sort_regions.m)
# ---------------------------------------------------------------------------

def _get_image_dimensions(image_path: str) -> tuple[int, int]:
    """Return (height, width) of a slide image."""
    if openslide is not None:
        slide = openslide.OpenSlide(image_path)
        w, h = slide.dimensions
        slide.close()
        return h, w
    else:
        from PIL import Image
        im = Image.open(image_path)
        w, h = im.size
        im.close()
        return h, w


def compute_pixel_regions(
    xy: list[list[tuple[float, float]]],
    nuc_svs_path: str,
    buff: int = 2000,
) -> dict:
    """Convert ROI vertex coordinates to buffered pixel regions and crop info.

    Uses min/max of all vertices (works for rectangles and arbitrary polygons).

    Args:
        xy: List of regions from parse_xml().
        nuc_svs_path: Path to nuclei SVS (used to get image dimensions).
        buff: Pixel buffer to add around each ROI.

    Returns:
        Dict with keys:
        - pixel_region_buff: list of ((row_start, row_stop), (col_start, col_stop))
        - cropregion: list of (x_offset, y_offset, width, height) for cropping
        - nm: list of ROI names ("ROI01", "ROI02", ...)
        - maxrow, maxcol: image dimensions
    """
    maxrow, maxcol = _get_image_dimensions(nuc_svs_path)

    pixel_region_buff = []
    cropregion = []
    nm = []

    for i, region in enumerate(xy, start=1):
        xs = [p[0] for p in region]
        ys = [p[1] for p in region]

        col_start = int(min(xs))
        col_stop = int(max(xs))
        row_start = int(min(ys))
        row_stop = int(max(ys))

        # Add buffer, clamp to image boundaries
        buf_row_start = max(0, row_start - buff)
        buf_row_stop = min(maxrow, row_stop + buff)
        buf_col_start = max(0, col_start - buff)
        buf_col_stop = min(maxcol, col_stop + buff)

        pixel_region_buff.append(
            ((buf_row_start, buf_row_stop), (buf_col_start, buf_col_stop))
        )

        # Crop offset = distance from buffer edge to actual ROI edge
        crop_x = col_start - buf_col_start
        crop_y = row_start - buf_row_start
        crop_w = col_stop - col_start
        crop_h = row_stop - row_start
        cropregion.append((crop_x, crop_y, crop_w, crop_h))

        nm.append(f'ROI{i:02d}')

    return {
        'pixel_region_buff': pixel_region_buff,
        'cropregion': cropregion,
        'nm': nm,
        'maxrow': maxrow,
        'maxcol': maxcol,
    }


# ---------------------------------------------------------------------------
# Image Reading
# ---------------------------------------------------------------------------

def read_region(svs_path: str, pixel_region) -> np.ndarray:
    """Read a buffered region from an SVS file.

    Args:
        svs_path: Path to the SVS file.
        pixel_region: ((row_start, row_stop), (col_start, col_stop))

    Returns:
        RGB numpy array (H, W, 3), dtype uint8.
    """
    (row_start, row_stop), (col_start, col_stop) = pixel_region
    width = col_stop - col_start
    height = row_stop - row_start

    if openslide is not None:
        slide = openslide.OpenSlide(svs_path)
        # OpenSlide uses (x, y) = (col, row) ordering
        region = slide.read_region((col_start, row_start), 0, (width, height))
        slide.close()
        return np.array(region)[:, :, :3]  # Drop alpha channel
    else:
        from PIL import Image
        im = Image.open(svs_path)
        cropped = im.crop((col_start, row_start, col_stop, row_stop))
        return np.array(cropped)[:, :, :3]


# ---------------------------------------------------------------------------
# TIFF Writing
# ---------------------------------------------------------------------------

def write_tiff(path: str, image: np.ndarray, tile=(240, 240), compression='jpeg'):
    """Write an RGB image as a tiled TIFF.

    Args:
        path: Output file path.
        image: RGB numpy array (H, W, 3), dtype uint8.
        tile: Tile dimensions.
        compression: Compression method.
    """
    if tifffile is None:
        raise ImportError('tifffile is required for TIFF writing')
    tifffile.imwrite(
        path, image,
        compression=compression,
        tile=tile,
        photometric='rgb',
        software='Python',
    )
