# Traffic Sign Detection Pipeline

This project implements a traditional Computer Vision pipeline for traffic sign detection using OpenCV.

The pipeline consists of:

1. ROI Extraction
2. HSV Color Filtering
3. Morphology + Area Filtering
4. Contour Extraction

---

# Project Structure

```text
traffic_sign_alignment/

├── input_images/
│ 
├── output_images/
│ 
├── standalone_demo/
│   ├── test_step01_roi.py
│   ├── test_step02_hsv.py
│   ├── test_step03_morphology.py
│   └── test_step04_contour.py
│ 
├── steps/
│   ├── step01_roi.py
│   ├── step02_hsv.py
│   ├── step03_morphology.py
│   └── step04_contour.py
├── utils/
│   ├── io.py
├── README.md
├── config.py
├── .gitignore
└── main.py
```

---

# Installation

Install required libraries:

```bash
pip install opencv-python matplotlib numpy
```

---

# STEP 1 — ROI Extraction

Purpose:

- Reduce unnecessary image regions
- Focus on likely traffic sign areas
- Improve processing efficiency

Method:

- Crop the right side of the image
- Remove lower irrelevant regions

Output:

- ROI image

---

# STEP 2 — HSV Color Filtering

Purpose:

- Detect traffic sign colors
- Separate signs from background

Method:

- Convert BGR image to HSV
- Detect:
  - Blue regions
  - Red regions

Output:

- Blue binary mask
- Red binary mask

---

# STEP 3 — Morphology + Area Filtering

Purpose:

- Clean noisy masks
- Fill small holes
- Remove tiny blobs

Method:

1. Morphological Closing
2. Connected Component Analysis
3. Area Filtering

Important:

Blue and red masks are processed separately before combining.

Pipeline:

```text
Blue Mask
→ Morphology
→ Area Filtering

Red Mask
→ Morphology
→ Area Filtering

Final Mask
= Blue + Red
```

Output:

- Clean binary mask

---

# STEP 4 — Contour Extraction

Purpose:

- Detect potential traffic sign objects

Method:

1. Find contours
2. Apply contour filtering:
   - Minimum area
   - Position filtering
   - Aspect ratio filtering
3. Draw:
   - Contours
   - Bounding boxes

Output:

- Final detection visualization

---

# How to Run

Run each testing file individually:

```bash
python test_step01_roi.py
```

```bash
python test_step02_hsv.py
```

```bash
python test_step03_morphology.py
```

```bash
python test_step04_contour.py
```

---

# Output

Generated outputs are saved inside:

```text
output_images/
```

The pipeline produces:

- ROI visualizations
- HSV masks
- Morphology results
- Filtered masks
- Contour detections
- Bounding boxes

---

# Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib

---

# Notes

This project uses a traditional Computer Vision approach instead of Deep Learning.

Detection quality depends on:

- Lighting conditions
- HSV thresholds
- Morphology parameters
- Camera angle
- Traffic sign visibility

This project is intended for educational and experimental purposes.