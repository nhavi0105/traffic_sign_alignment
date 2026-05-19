import cv2


# ===================================================
# STEP 4 — CONTOUR EXTRACTION
# ===================================================

def extract_contours(mask, roi):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    output = roi.copy()

    detected = 0

    roi_h, roi_w = roi.shape[:2]

    for cnt in contours:

        area = cv2.contourArea(cnt)

        # ------------------------------------------------
        # AREA FILTER
        # ------------------------------------------------

        if area < 250:
            continue

        # ------------------------------------------------
        # BOUNDING BOX
        # ------------------------------------------------

        x, y, w, h = cv2.boundingRect(cnt)

        if h == 0:
            continue

        aspect_ratio = w / float(h)

        # ------------------------------------------------
        # POSITION FILTER
        # ------------------------------------------------

        center_x = x + w // 2
        center_y = y + h // 2

        if center_y > roi_h * 0.90:
            continue

        if center_x < roi_w * 0.10:
            continue

        # ------------------------------------------------
        # REMOVE THIN OBJECTS
        # ------------------------------------------------

        if aspect_ratio < 0.35:
            continue

        # ------------------------------------------------
        # DRAW CONTOUR
        # ------------------------------------------------

        detected += 1

        cv2.drawContours(
            output,
            [cnt],
            -1,
            (0, 255, 0),
            2
        )

        # ------------------------------------------------
        # DRAW BOUNDING BOX
        # ------------------------------------------------

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    return output, detected