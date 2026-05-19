# config.py
# Configuration settings for the Traffic Sign Alignment Pipeline

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(PROJECT_ROOT, 'input_images', 'images')
LABELS_DIR = os.path.join(PROJECT_ROOT, 'input_images', 'labels')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output_images')

# Processing parameters
DEFAULT_IMAGE_SIZE = (224, 224)
SHARPEN_FACTOR = 1.5
ENHANCE_CONTRAST = 1.2

# Pipeline steps
ENABLED_STEPS = [
    'roi',
    'hsv_filter',
    'morphology',
    'edge_contour',
    'shape_analysis',
    'rectify',
    'enhance'
]