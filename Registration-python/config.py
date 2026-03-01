# Feature detection
FEATURE_DETECTOR = "SIFT"       # "SIFT" or "SURF"
MAX_FEATURES = 50000

# Feature matching
MATCH_RATIO = 0.8               # Lowe's ratio test threshold

# RANSAC parameters
RANSAC_MAX_TRIALS_PRIMARY = 100000
RANSAC_MAX_TRIALS_FALLBACK = 50000
RANSAC_CONFIDENCE = 0.96
RANSAC_MAX_DISTANCE = 5.0

# Dual-channel fallback
FALLBACK_KP_THRESHOLD = 5      # Try second channel if <= this many inliers

# Region processing
BUFFER_SIZE = 2000              # Pixels around ROI

# Output
CHECK_IMAGE_SCALE = 0.0625     # 6.25% for registration check images
TIFF_TILE_SIZE = (240, 240)
TIFF_COMPRESSION = 'jpeg'

# Skip list (markers not to register -- matched via substring)
SKIP_MARKERS = [
    'NUCLEI', 'HEM', 'HEMATOXYLIN',
    'FIRSTHEMA', 'FIRSTH', 'FIRSTHEM1', 'SECONDHEM',
]
