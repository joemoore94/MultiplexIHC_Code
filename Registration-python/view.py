"""Quick TIFF viewer for registration output.

Usage:
    python view.py /path/to/slide_dir
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_triples(slide_dir: Path) -> list[tuple[object, object, object, str]]:
    """Find matching (HEM, raw, registered) image triplets from slide folder structure.

    Accepts either a slide directory (containing Registered_Regions/) or a
    parent directory (containing slide subdirectories).
    """
    triples = []

    reg_root = slide_dir / 'Registered_Regions'
    raw_root = slide_dir / 'Raw_Regions'

    # If no Registered_Regions here, look one level deeper
    if not reg_root.exists():
        for sub in sorted(slide_dir.iterdir()):
            if sub.is_dir() and (sub / 'Registered_Regions').exists():
                triples.extend(load_triples(sub))
        return triples

    for roi_dir in sorted(reg_root.iterdir()):
        if not roi_dir.is_dir():
            continue
        roi_name = roi_dir.name

        hem_files = sorted(roi_dir.glob(f'NUCLEI_*_{roi_name}.tif'))
        hem_img = mpimg.imread(str(hem_files[0])) if hem_files else None

        raw_dir = raw_root / roi_name
        for reg_file in sorted(roi_dir.glob(f'reg_*_{roi_name}.tif')):
            key = reg_file.stem[len('reg_'):]
            raw_file = raw_dir / f'raw_{key}.tif' if raw_dir.exists() else None

            reg_img = mpimg.imread(str(reg_file))
            raw_img = mpimg.imread(str(raw_file)) if raw_file and raw_file.exists() else None

            triples.append((hem_img, raw_img, reg_img, key))

    return triples


def show_triple(triples: list[tuple[object, object, object, str]]):
    if not triples:
        print('No matching image triplets found.')
        return

    count = len(triples)
    fig, (ax_h, ax_raw, ax_reg) = plt.subplots(1, 3, figsize=(24, 8))
    idx = [0]

    def draw(i):
        hem_img, raw_img, reg_img, label = triples[i]
        ax_h.clear()
        ax_raw.clear()
        ax_reg.clear()
        if hem_img is not None:
            ax_h.imshow(hem_img)
        ax_h.set_title('HEM (reference)', fontsize=9)
        ax_h.axis('off')
        if raw_img is not None:
            ax_raw.imshow(raw_img)
        ax_raw.set_title('Unregistered', fontsize=9)
        ax_raw.axis('off')
        ax_reg.imshow(reg_img)
        ax_reg.set_title('Registered', fontsize=9)
        ax_reg.axis('off')
        fig.suptitle(f'{label}  [{i + 1}/{count}]  —  arrow keys to navigate  |  q to quit', fontsize=10)
        fig.canvas.draw()

    def on_key(event):
        if event.key in ('right', 'n', ' '):
            idx[0] = (idx[0] + 1) % count
        elif event.key in ('left', 'p', 'b'):
            idx[0] = (idx[0] - 1) % count
        elif event.key == 'q':
            plt.close()
            return
        draw(idx[0])

    fig.canvas.mpl_connect('key_press_event', on_key)
    draw(0)
    if count > 1:
        print(f'{count} markers — arrow keys or space/b to navigate, q to quit')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    show_triple(load_triples(Path(sys.argv[1])))
