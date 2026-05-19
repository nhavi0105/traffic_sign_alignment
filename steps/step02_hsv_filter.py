# steps/step02_hsv_filter.py

import cv2
import numpy as np


# ---------------------------------------------------
# STEP 2 — HSV FILTERING
# Using:
# - HSV color space
# - Gaussian blur
# - Blue sign detection
# - Red sign detection
# ---------------------------------------------------

def hsv_filter(roi):

    # ------------------------------------------------
    # Blur to stabilize HSV
    # ------------------------------------------------

    blurred = cv2.GaussianBlur(
        roi,
        (5, 5),
        0
    )

    # ------------------------------------------------
    # Convert BGR → HSV
    # ------------------------------------------------

    hsv = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2HSV
    )

    # ------------------------------------------------
    # BLUE MASK
    # ------------------------------------------------

    lower_blue = np.array([100, 120, 70])
    upper_blue = np.array([130, 255, 255])

    blue_mask = cv2.inRange(
        hsv,
        lower_blue,
        upper_blue
    )

    # ------------------------------------------------
    # RED MASK
    # ------------------------------------------------

    lower_red1 = np.array([0, 140, 80])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 140, 80])
    upper_red2 = np.array([180, 255, 255])

    red_mask1 = cv2.inRange(
        hsv,
        lower_red1,
        upper_red1
    )

    red_mask2 = cv2.inRange(
        hsv,
        lower_red2,
        upper_red2
    )

    # ------------------------------------------------
    # Combine red masks
    # ------------------------------------------------

    red_mask = cv2.add(
        red_mask1,
        red_mask2
    )

    return blue_mask, red_mask