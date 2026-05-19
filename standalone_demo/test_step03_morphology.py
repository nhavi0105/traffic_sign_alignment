# steps/step03_morphology.py

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


# ---------------------------------------------------
# STEP 1 — SIMPLE HEURISTIC ROI
# ---------------------------------------------------

def extract_roi(image):

    h, w = image.shape[:2]

    x_start = int(w * 0.50)

    y_start = 0
    y_end = int(h * 0.85)

    roi = image[
        y_start:y_end,
        x_start:w
    ]

    return roi


# ---------------------------------------------------
# STEP 2 — HSV FILTERING
# ---------------------------------------------------

def hsv_filter(roi):

    # ------------------------------------------------
    # Blur before HSV
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

    red_mask = cv2.add(
        red_mask1,
        red_mask2
    )

    return blue_mask, red_mask


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

    return closed, filtered


# ---------------------------------------------------
# CREATE OUTPUT FOLDER
# ---------------------------------------------------

os.makedirs(
    'output_images',
    exist_ok=True
)

input_folder = 'input_images'

image_files = os.listdir(input_folder)


# ---------------------------------------------------
# PROCESS ALL IMAGES
# ---------------------------------------------------

for file_name in image_files:

    image_path = os.path.join(
        input_folder,
        file_name
    )

    img = cv2.imread(image_path)

    if img is None:

        print(f"Cannot load image: {file_name}")
        continue

    # ------------------------------------------------
    # STEP 1 — ROI
    # ------------------------------------------------

    roi = extract_roi(img)

    # ------------------------------------------------
    # STEP 2 — HSV FILTERING
    # ------------------------------------------------

    blue_mask, red_mask = hsv_filter(roi)

    # ------------------------------------------------
    # STEP 3 — MORPHOLOGY means we will have two separate masks:
    # one for blue and one for red.
    # 
    # Process each color separately
    # ------------------------------------------------

    blue_closed, blue_filtered = morphology_filter(
        blue_mask
    )

    red_closed, red_filtered = morphology_filter(
        red_mask
    )

    # ------------------------------------------------
    # FINAL COMBINED MASK
    # ------------------------------------------------

    filtered_mask = cv2.add(
        blue_filtered,
        red_filtered
    )

    # ------------------------------------------------
    # SAVE OUTPUTS
    # ------------------------------------------------

    cv2.imwrite(
        f'output_images/blue_closed_{file_name}',
        blue_closed
    )

    cv2.imwrite(
        f'output_images/red_closed_{file_name}',
        red_closed
    )

    cv2.imwrite(
        f'output_images/blue_filtered_{file_name}',
        blue_filtered
    )

    cv2.imwrite(
        f'output_images/red_filtered_{file_name}',
        red_filtered
    )

    cv2.imwrite(
        f'output_images/final_filtered_{file_name}',
        filtered_mask
    )

    # ------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------

    titles = [
        'ROI',
        'Blue Mask',
        'Red Mask',
        'Blue After Morphology',
        'Red After Morphology',
        'Final Combined Mask'
    ]

    images = [
        cv2.cvtColor(roi, cv2.COLOR_BGR2RGB),
        blue_mask,
        red_mask,
        blue_filtered,
        red_filtered,
        filtered_mask
    ]

    plt.figure(figsize=(24, 5))

    for i in range(6):

        plt.subplot(1, 6, i + 1)

        if i == 0:

            plt.imshow(images[i])

        else:

            plt.imshow(
                images[i],
                cmap='gray'
            )

        plt.title(titles[i])

        plt.axis('off')

    plt.tight_layout()

    plt.show()

    print(f"Processed: {file_name}")