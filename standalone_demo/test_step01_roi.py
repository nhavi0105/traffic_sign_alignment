import os
import cv2
from matplotlib import pyplot as plt


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

# ---------------------------------------------------
# CREATE OUTPUT FOLDER
# ---------------------------------------------------

os.makedirs('output_images', exist_ok=True)

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

        print(f"Không load được ảnh: {file_name}")
        continue

    # ---------------------------------------------
    # Run ROI extraction
    # ---------------------------------------------
    roi = extract_roi(img)

    # ---------------------------------------------
    # Save ROI output
    # ---------------------------------------------
    output_path = os.path.join(
        'output_images',
        f'roi_{file_name}'
    )

    cv2.imwrite(output_path, roi)

    # ---------------------------------------------
    # Visualization
    # ---------------------------------------------
    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)

    plt.imshow(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )

    plt.title("Original")

    plt.axis('off')

    # ROI image
    plt.subplot(1, 2, 2)

    plt.imshow(
        cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    )

    plt.title("Simple Heuristic ROI")

    plt.axis('off')

    plt.tight_layout()

    plt.show()

    print(f"Processed: {file_name}")