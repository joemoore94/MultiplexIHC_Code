#!/bin/bash
# mIHC Registration Pipeline — one-time environment setup
# Usage: ./setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# 1. Install homebrew dependencies (openslide + opencv with nonfree/SURF)
# ---------------------------------------------------------------------------
if ! brew list openslide &>/dev/null; then
    echo "Installing openslide via Homebrew..."
    brew install openslide
else
    echo "openslide already installed."
fi

if ! brew list opencv &>/dev/null; then
    echo "Installing opencv via Homebrew (includes SURF)..."
    brew install opencv
else
    echo "opencv already installed."
fi

# ---------------------------------------------------------------------------
# 2. Create virtual environment with Python 3.14 (if missing)
# ---------------------------------------------------------------------------
VENV="$SCRIPT_DIR/.venv"
PYTHON="$(brew --prefix)/bin/python3.14"

if [ ! -x "$PYTHON" ]; then
    echo "Error: python3.14 not found at $PYTHON"
    echo "It should have been installed as a dependency of opencv."
    exit 1
fi

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment with Python 3.14..."
    "$PYTHON" -m venv "$VENV"
fi

source "$VENV/bin/activate"
pip install --quiet --upgrade pip

# ---------------------------------------------------------------------------
# 3. Install Python dependencies (excluding opencv — provided by Homebrew)
# ---------------------------------------------------------------------------
if ! python -c "import tifffile, numpy, openslide" &>/dev/null; then
    echo "Installing Python dependencies..."
    pip install --quiet tifffile openslide-python openslide-bin imagecodecs matplotlib numpy
else
    echo "Python dependencies already installed."
fi

# ---------------------------------------------------------------------------
# 4. Link Homebrew cv2 into the venv via a .pth file
# ---------------------------------------------------------------------------
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
PTH="$SITE/homebrew-cv2.pth"
BREW_CV2="$(brew --prefix)/lib/python3.14/site-packages"

if [ ! -f "$PTH" ]; then
    echo "Linking Homebrew cv2 (SURF enabled) into venv..."
    echo "$BREW_CV2" > "$PTH"
fi

# Verify SURF
python -c "import cv2; cv2.xfeatures2d.SURF_create(400)" && echo "SURF available." || echo "Warning: SURF not available."

echo ""
echo "Setup complete. Run the pipeline with:"
echo "  ./mihc.sh /path/to/slides"
