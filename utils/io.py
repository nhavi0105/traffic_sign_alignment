# utils/io.py
# Input/Output utilities for the Traffic Sign Alignment Pipeline

import os
import cv2

from config import INPUT_DIR, OUTPUT_DIR


# ---------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------
def load_image(image_path):
    """
    Load an image from file.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}"
        )

    return image


# ---------------------------------------------------
# SAVE IMAGE
# ---------------------------------------------------
def save_image(image, filename, output_dir=OUTPUT_DIR):
    """
    Save an image to output directory.
    """

    # Create output folder if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(
        output_dir,
        filename
    )

    cv2.imwrite(output_path, image)

    return output_path


# ---------------------------------------------------
# GET IMAGE LIST
# ---------------------------------------------------
def get_image_list(input_dir=INPUT_DIR):
    """
    Get list of image paths from input directory.
    """

    if not os.path.exists(input_dir):
        return []

    extensions = [
        '.jpg',
        '.jpeg',
        '.png',
        '.bmp'
    ]

    image_list = []

    for file in sorted(os.listdir(input_dir)):

        if any(
            file.lower().endswith(ext)
            for ext in extensions
        ):

            image_path = os.path.join(
                input_dir,
                file
            )

            image_list.append(image_path)

    return image_list