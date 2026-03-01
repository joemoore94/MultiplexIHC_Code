"""Main entry point for the Python mIHC registration pipeline.

Usage:
    python register.py /path/to/parent_dir
    python register.py /path/to/parent_dir --detector SURF
    python register.py /path/to/parent_dir --step
"""
import argparse
import datetime
import logging
from pathlib import Path
from typing import Optional

import config
from lib.features import create_detector, register_single_marker
from lib.io import (
    compute_pixel_regions,
    get_image_sets,
    parse_xml,
    read_region,
)
from lib.transform import (
    crop_to_roi,
    save_failed,
    save_nuclei_reference,
    save_raw,
    save_registered,
    warp_image,
)

logger = logging.getLogger(__name__)


def is_skipped(marker_name: str) -> bool:
    """Check if a marker should be skipped (nuclei, hematoxylin, etc.)."""
    return any(skip in marker_name for skip in config.SKIP_MARKERS)


def is_already_done(roi_dir: Path, filename: str, roi_name: str) -> bool:
    """Check if a marker has already been registered or marked as failed."""
    reg_path = roi_dir / f'reg_{filename}_{roi_name}.tif'
    nonreg_path = roi_dir / f'reg_NONREG_{filename}_{roi_name}.tif'
    return reg_path.exists() or nonreg_path.exists()


def register_slide(
    slide_dir: Path,
    check_dir: Path,
    detector_method: Optional[str] = None,
    step: bool = False,
):
    """Register all markers for all ROIs in a single slide directory.

    Args:
        slide_dir: Directory containing .svs and .xml files for one slide.
        check_dir: Parent-level Registration_Check directory.
        detector_method: "SIFT" or "SURF" (or None for config default).
    """
    svs_paths, xml_path, filenames, nuc_index = get_image_sets(str(slide_dir))

    if not svs_paths or not xml_path:
        logger.warning(f'No SVS or XML files found in {slide_dir.name}, skipping.')
        return

    logger.info(f'{len(svs_paths)} SVS files found in {slide_dir.name}')

    roi_coords = parse_xml(xml_path)
    regions = compute_pixel_regions(
        roi_coords, svs_paths[nuc_index], buff=config.BUFFER_SIZE
    )

    detector = create_detector(detector_method)
    nuc_filename = filenames[nuc_index]

    for roi_idx, roi_name in enumerate(regions['nm']):
        pixel_region = regions['pixel_region_buff'][roi_idx]
        cropregion = regions['cropregion'][roi_idx]

        # Read nuclei reference for this ROI and save it immediately
        nuc_ref = read_region(svs_paths[nuc_index], pixel_region)
        nuc_crop = crop_to_roi(nuc_ref, cropregion)

        roi_dir = slide_dir / 'Registered_Regions' / roi_name
        roi_dir.mkdir(parents=True, exist_ok=True)

        save_nuclei_reference(nuc_crop, nuc_filename, roi_name, slide_dir, check_dir)

        # Register each marker against the nuclei reference
        for svs_path, filename in zip(svs_paths, filenames):
            if is_skipped(filename):
                continue
            if is_already_done(roi_dir, filename, roi_name):
                logger.debug(f'Skipping {filename} {roi_name} (already done)')
                continue

            logger.info(f'Processing {filename} {roi_name}...')

            try:
                obj_rgb = read_region(svs_path, pixel_region)
                save_raw(crop_to_roi(obj_rgb, cropregion), filename, roi_name, slide_dir)
                M, status = register_single_marker(nuc_ref, obj_rgb, detector)

                if status == "success" and M is not None:
                    warped = warp_image(obj_rgb, M)
                    cropped = crop_to_roi(warped, cropregion)
                    save_registered(cropped, filename, roi_name, slide_dir, check_dir)
                else:
                    logger.warning(
                        f'{filename} {roi_name}: registration {status}'
                    )
                    save_failed(
                        obj_rgb, nuc_ref, filename, nuc_filename,
                        roi_name, slide_dir,
                    )
            except Exception:
                logger.exception(f'{filename} {roi_name}: unexpected error, skipping')

            if step:
                reply = input('  Press Enter to continue, or type "exit" to stop: ').strip().lower()
                if reply == 'exit':
                    logger.info('Step mode: user exited.')
                    return

        logger.info(f'{roi_name} registration complete!')


def main(parent_dir: str, detector: Optional[str] = None, step: bool = False):
    """Run registration on all slides in a parent directory."""
    parent = Path(parent_dir)
    check_dir = parent / 'Registration_Check'
    check_dir.mkdir(exist_ok=True)

    for slide_dir in sorted(parent.iterdir()):
        if not slide_dir.is_dir():
            continue
        if slide_dir.name in ('Registration_Check', '.DS_Store'):
            continue

        logger.info(f'Slide: {slide_dir.name}')
        register_slide(slide_dir, check_dir, detector_method=detector, step=step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mIHC Registration Pipeline')
    parser.add_argument('parent_dir', help='Directory containing all slide folders')
    parser.add_argument(
        '--detector', choices=['SIFT', 'SURF'],
        default=config.FEATURE_DETECTOR,
        help='Feature detector (default: %(default)s)',
    )
    parser.add_argument(
        '--step', action='store_true',
        help='Pause for confirmation after each marker is registered',
    )
    args = parser.parse_args()

    log_level = logging.DEBUG
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    # Log to terminal
    logging.basicConfig(format=log_format, level=log_level)

    # Log to file inside the Registration-python directory
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(log_dir / f'registration_{timestamp}.log')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)

    main(args.parent_dir, detector=args.detector, step=args.step)
