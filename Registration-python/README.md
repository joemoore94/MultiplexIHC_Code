# Registration (Python)

Python port of the MATLAB mIHC registration pipeline. See
`MATLAB_TO_PYTHON_REFACTORING_STRATEGY.md` for full details.

## Quick start

1. Install system dependency (macOS):

```bash
brew install openslide
```

2. Create a virtual environment and install:

```bash
cd python/
python -m venv .venv
source .venv/bin/activate
pip install .             # core dependencies
pip install .[dev]        # adds ruff + mypy
pip install .[test]       # adds pytest + scikit-image
pip install .[surf]       # adds SURF support (opencv-contrib-python)
```

3. Run registration:

```bash
python register.py /path/to/parent_dir
python register.py /path/to/parent_dir --detector SURF   # optional: use SURF
python register.py /path/to/parent_dir -v                # verbose logging
```

4. Run tests and static checks:

```bash
pytest tests/
ruff check .
mypy .
```

## Structure

```
python/
├── register.py              # Main entry point + CLI
├── registration/
│   ├── __init__.py
│   ├── io.py                # XML parsing, SVS reading, TIFF writing, file discovery
│   ├── features.py          # SIFT/SURF detection, matching, transform estimation
│   └── transform.py         # Warping, cropping, output generation
├── tests/
│   ├── test_io.py
│   ├── test_features.py
│   └── test_end_to_end.py
├── config.py                # All tunable parameters
└── pyproject.toml           # Dependencies, Python version, tool config
```
