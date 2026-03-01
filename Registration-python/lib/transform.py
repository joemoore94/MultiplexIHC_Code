"""Image warping, cropping, and output saving.

Handles applying transforms, writing registered TIFFs, generating QC images,
and routing failed registrations to Redo folders.
"""
import logging
from pathlib import Path

import cv2
import numpy as np

import config

from .io import write_tiff

logger = logging.getLogger(__name__)


def warp_image(image: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine transform to an RGB image.

    Each channel is warped independently, matching the MATLAB per-channel
    imwarp approach.

    Args:
        image: RGB image (H, W, 3), uint8.
        M: 2x3 affine matrix from estimateAffinePartial2D.

    Returns:
        Warped RGB image, same shape as input.
    """
    h, w = image.shape[:2]
    warped_channels = []
    for c in range(image.shape[2]):
        warped = cv2.warpAffine(
            image[:, :, c], M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_channels.append(warped)
    return np.stack(warped_channels, axis=2)


def crop_to_roi(image: np.ndarray, cropregion: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a warped image back to the original ROI boundaries.

    Args:
        image: Warped image (H, W, 3).
        cropregion: (x_offset, y_offset, width, height) from compute_pixel_regions.

    Returns:
        Cropped image.
    """
    x, y, w, h = cropregion
    return image[y:y + h, x:x + w]


def save_registered(
    registered: np.ndarray,
    filename: str,
    roi_name: str,
    slide_dir: Path,
    check_dir: Path,
):
    """Save a successfully registered image and its low-res QC check.

    Args:
        registered: Cropped registered image (H, W, 3), uint8.
        filename: UPPERCASE marker name (e.g. "CD4_SLIDE_001").
        roi_name: ROI identifier (e.g. "ROI01").
        slide_dir: Slide-level directory (contains Registered_Regions/).
        check_dir: Parent-level Registration_Check directory.
    """
    roi_dir = slide_dir / 'Registered_Regions' / roi_name
    roi_dir.mkdir(parents=True, exist_ok=True)

    out_path = roi_dir / f'reg_{filename}_{roi_name}.tif'
    write_tiff(str(out_path), registered)
    logger.info(f'{roi_name} {filename} image registered!')

    # Low-res registration check image
    check_dir.mkdir(parents=True, exist_ok=True)
    scale = config.CHECK_IMAGE_SCALE
    h, w = registered.shape[:2]
    small = cv2.resize(registered, (int(w * scale), int(h * scale)))
    check_path = check_dir / f'regck_{filename}_{roi_name}.tif'
    write_tiff(str(check_path), small)


def save_nuclei_reference(
    nuc_crop: np.ndarray,
    nuc_filename: str,
    roi_name: str,
    slide_dir: Path,
    check_dir: Path,
):
    """Save the cropped nuclei reference for a ROI.

    Args:
        nuc_crop: Cropped nuclei image (H, W, 3), uint8.
        nuc_filename: UPPERCASE nuclei filename (e.g. "NUCLEI_DAPI_SLIDE_001").
        roi_name: ROI identifier.
        slide_dir: Slide-level directory.
        check_dir: Registration_Check directory.
    """
    roi_dir = slide_dir / 'Registered_Regions' / roi_name
    roi_dir.mkdir(parents=True, exist_ok=True)

    out_path = roi_dir / f'NUCLEI_{nuc_filename}_{roi_name}.tif'
    write_tiff(str(out_path), nuc_crop)

    scale = config.CHECK_IMAGE_SCALE
    h, w = nuc_crop.shape[:2]
    small = cv2.resize(nuc_crop, (int(w * scale), int(h * scale)))
    check_path = check_dir / f'NUCLEIck_{nuc_filename}_{roi_name}.tif'
    write_tiff(str(check_path), small)


def save_raw(
    raw: np.ndarray,
    filename: str,
    roi_name: str,
    slide_dir: Path,
):
    """Save the raw (unregistered) cropped marker image.

    Args:
        raw: Cropped but unregistered marker image (H, W, 3), uint8.
        filename: UPPERCASE marker name.
        roi_name: ROI identifier.
        slide_dir: Slide-level directory.
    """
    roi_dir = slide_dir / 'Raw_Regions' / roi_name
    roi_dir.mkdir(parents=True, exist_ok=True)

    out_path = roi_dir / f'raw_{filename}_{roi_name}.tif'
    write_tiff(str(out_path), raw)


def save_failed(
    obj_rgb: np.ndarray,
    nuc_rgb: np.ndarray,
    filename: str,
    nuc_filename: str,
    roi_name: str,
    slide_dir: Path,
):
    """Save a failed registration to the Redo folder.

    Saves both the unregistered marker image and the nuclei reference so the
    user can attempt manual registration.

    Args:
        obj_rgb: Unregistered marker image (H, W, 3), uint8.
        nuc_rgb: Nuclei reference image (H, W, 3), uint8.
        filename: UPPERCASE marker name.
        nuc_filename: UPPERCASE nuclei filename.
        roi_name: ROI identifier.
        slide_dir: Slide-level directory.
    """
    redo_dir = slide_dir / f'Redo_{roi_name}'
    redo_dir.mkdir(parents=True, exist_ok=True)

    nonreg_path = redo_dir / f'nonreg_{filename}_{roi_name}.tif'
    write_tiff(str(nonreg_path), obj_rgb)
    logger.warning(f'{filename} was not automatically registered for {roi_name}.')

    nuc_path = redo_dir / f'NUCLEI_{nuc_filename}_{roi_name}.tif'
    if not nuc_path.exists():
        write_tiff(str(nuc_path), nuc_rgb)
