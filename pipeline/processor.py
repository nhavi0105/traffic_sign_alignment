# pipeline/processor.py
# Main processor for the Traffic Sign Alignment Pipeline

import os

from config import ENABLED_STEPS
from utils.io import (
    load_image,
    save_image,
    get_image_list,
    load_label
)

# Import pipeline steps
from steps.step01_roi import extract_roi
from steps.step02_hsv_filter import hsv_filter
from steps.step03_morphology import apply_morphology
from steps.step04_contour import detect_contours
from steps.step05_shape_analysis import analyze_shapes
from steps.step06_rectify import rectify_sign
from steps.step07_enhance import enhance_image


class TrafficSignProcessor:

    def __init__(self):

        # Pipeline step mapping
        self.steps = {
            'roi': self._roi_step,
            'hsv_filter': self._hsv_filter_step,
            'morphology': self._morphology_step,
            'edge_contour': self._edge_contour_step,
            'shape_analysis': self._shape_analysis_step,
            'rectify': self._rectify_step,
            'enhance': self._enhance_step
        }

    # ---------------------------------------------------
    # STEP 1 — ROI Extraction
    # ---------------------------------------------------
    def _roi_step(self, image):

        roi = extract_roi(image)
        return roi

    # ---------------------------------------------------
    # STEP 2 — HSV Filtering
    # ---------------------------------------------------
    def _hsv_filter_step(self, image):

        mask = hsv_filter(image)
        return mask

    # ---------------------------------------------------
    # STEP 3 — Morphology
    # ---------------------------------------------------
    def _morphology_step(self, mask):

        cleaned_mask = apply_morphology(mask)
        return cleaned_mask

    # ---------------------------------------------------
    # STEP 4 — Edge + Contour Detection
    # ---------------------------------------------------
    def _edge_contour_step(self, mask):

        contours = detect_contours(mask)
        return contours

    # ---------------------------------------------------
    # STEP 5 — Shape Analysis
    # ---------------------------------------------------
    def _shape_analysis_step(self, contours):

        corners = analyze_shapes(contours)
        return corners

    # ---------------------------------------------------
    # STEP 6 — Perspective Rectification
    # ---------------------------------------------------
    def _rectify_step(self, data):

        # data should contain:
        # original image + corner points

        original_image = data["image"]
        corners = data["corners"]

        rectified = rectify_sign(original_image, corners)

        return rectified

    # ---------------------------------------------------
    # STEP 7 — Enhancement
    # ---------------------------------------------------
    def _enhance_step(self, image):

        enhanced = enhance_image(image)
        return enhanced

    # ---------------------------------------------------
    # PROCESS SINGLE IMAGE
    # ---------------------------------------------------
    def process_image(self, image_path, label_path=None):

        # Load original image
        original_image = load_image(image_path)

        # Optional label loading
        labels = load_label(label_path)

        # Current working data
        current_data = original_image

        # Store intermediate outputs if needed
        pipeline_data = {
            "image": original_image
        }

        for step_name in ENABLED_STEPS:

            if step_name not in self.steps:
                continue

            # Run current step
            result = self.steps[step_name](current_data)

            # Store intermediate outputs
            pipeline_data[step_name] = result

            # Special handling for shape analysis
            if step_name == "shape_analysis":

                current_data = {
                    "image": original_image,
                    "corners": result
                }

            else:
                current_data = result

        # Final output image
        final_output = current_data

        # Save output
        filename = os.path.basename(image_path)

        output_path = save_image(
            final_output,
            f"processed_{filename}"
        )

        return output_path

    # ---------------------------------------------------
    # RUN FULL PIPELINE
    # ---------------------------------------------------
    def run_pipeline(self):

        pairs = get_image_list()

        if not pairs:
            print("No images found in input directory.")
            return

        print(f"Processing {len(pairs)} images...")

        for image_path, label_path in pairs:

            try:

                output_path = self.process_image(
                    image_path,
                    label_path
                )

                print(
                    f"Processed: {image_path} -> {output_path}"
                )

            except Exception as e:

                print(
                    f"Error processing {image_path}: {e}"
                )

        print("Pipeline completed.")