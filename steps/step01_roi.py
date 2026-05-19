# steps/step01_roi.py

import cv2


# ---------------------------------------------------
# STEP 1 — SIMPLE HEURISTIC ROI
# Using:
# - Right-side crop
# - Upper-region crop
# ---------------------------------------------------

def extract_roi(image):

    h, w = image.shape[:2]

    # ------------------------------------------------
    # ROI boundaries
    # ------------------------------------------------

    # Keep right-half region
    x_start = int(w * 0.50)

    # Keep most upper area
    y_start = 0
    y_end = int(h * 0.85)

    # ------------------------------------------------
    # Final ROI
    # ------------------------------------------------

    roi = image[
        y_start:y_end,
        x_start:w
    ]

    return roi