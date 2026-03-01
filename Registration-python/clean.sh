#!/bin/bash
# Remove all registration output TIFFs for a slide directory.
# Usage: ./clean.sh /path/to/slides [--yes]
#
# Deletes inside each slide subdirectory:
#   Registered_Regions/  Raw_Regions/  Redo_*/  Registration_Check/

set -e

if [ -z "$1" ]; then
    echo "Usage: ./clean.sh /path/to/slides [--yes]"
    exit 1
fi

TARGET="$1"
FORCE="${2:-}"

# Collect what would be deleted
DIRS=()
while IFS= read -r -d '' d; do
    DIRS+=("$d")
done < <(find "$TARGET" \( \
    -name "Registered_Regions" \
    -o -name "Raw_Regions" \
    -o -name "Registration_Check" \
    -o -name "Redo_*" \
    \) -type d -print0 2>/dev/null)

if [ ${#DIRS[@]} -eq 0 ]; then
    echo "Nothing to clean in $TARGET"
    exit 0
fi

echo "The following directories will be deleted:"
for d in "${DIRS[@]}"; do
    echo "  $d"
done

if [ "$FORCE" != "--yes" ]; then
    read -r -p "Proceed? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

for d in "${DIRS[@]}"; do
    rm -rf "$d"
    echo "Removed: $d"
done

echo "Done."
