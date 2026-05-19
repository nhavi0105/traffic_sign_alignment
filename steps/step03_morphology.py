# steps/step03_morphology.py

import cv2
import numpy as np


# ---------------------------------------------------
# STEP 3 — MORPHOLOGY + AREA FILTERING
# ---------------------------------------------------

def morphology_filter(mask):

    # ------------------------------------------------
    # MORPHOLOGICAL CLOSING
    # Fill small holes
    # ------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7)
    )

    closed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ------------------------------------------------
    # CONNECTED COMPONENT ANALYSIS
    # Remove tiny noise blobs
    # ------------------------------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed,
        connectivity=8
    )

    filtered = np.zeros_like(closed)

    # ------------------------------------------------
    # Keep only large enough regions
    # ------------------------------------------------

    MIN_AREA = 120

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area > MIN_AREA:

            filtered[labels == i] = 255

    return filtered