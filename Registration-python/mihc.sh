#!/bin/bash
# mIHC Registration Pipeline
# Usage: ./mihc.sh /path/to/slides

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

if [ -z "$1" ]; then
    echo "Usage: ./mihc.sh /path/to/slides"
    exit 1
fi

SLIDES="$1"

# Auto-setup if needed
if [ ! -d "$VENV" ]; then
    echo "First run — setting up environment..."
    bash "$SCRIPT_DIR/setup.sh"
fi

source "$VENV/bin/activate"

echo ""
echo "mIHC Pipeline  |  $SLIDES"
echo "────────────────────────────────────"
echo "  1) Register"
echo "  2) View results"
echo "  3) Clean output"
echo "  q) Quit"
echo ""
read -r -p "> " choice

case "$choice" in
    1)
        echo ""
        echo "Detector:  1) SIFT (default)  2) SURF"
        read -r -p "> " det
        case "$det" in
            2) DETECTOR="SURF" ;;
            *) DETECTOR="SIFT" ;;
        esac

        echo ""
        read -r -p "Step mode? [y/N]: " step
        STEP_FLAG=""
        [[ "$step" =~ ^[Yy]$ ]] && STEP_FLAG="--step"

        echo ""
        python "$SCRIPT_DIR/register.py" "$SLIDES" --detector "$DETECTOR" $STEP_FLAG
        ;;
    2)
        echo ""
        python "$SCRIPT_DIR/view.py" "$SLIDES"
        ;;
    3)
        echo ""
        bash "$SCRIPT_DIR/clean.sh" "$SLIDES"
        ;;
    q|Q)
        exit 0
        ;;
    *)
        echo "Invalid choice."
        exit 1
        ;;
esac
