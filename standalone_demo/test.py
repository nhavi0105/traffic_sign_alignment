def semantic_contour_detection(mask, roi_image):

    """
    Semantic contour detection pipeline for traffic signs.

    Pipeline:
    1. Extract contours from segmented mask
    2. Merge fragmented contours using probabilistic affinity
    3. Compute semantic/geometric features
    4. Filter unlikely candidates
    5. Return detected sign contours
    """

    # =========================================================
    # Similarity Functions
    # =========================================================

    def gaussian_similarity(distance, sigma):

        """
        Convert distance into smooth similarity score.

        close distance -> similarity near 1
        far distance -> similarity near 0
        """

        return np.exp(
            -(distance ** 2) / (2.0 * sigma ** 2)
        )

    def aspect_similarity(ar1, ar2):

        """
        Compare contour aspect ratios.
        """

        if ar1 <= 0 or ar2 <= 0:
            return 0.0

        ratio = abs(ar1 - ar2) / max(ar1, ar2)

        return np.exp(
            -(ratio ** 2) / (2.0 * 0.5 ** 2)
        )

    def orientation_similarity(theta1, theta2):

        """
        Compare contour orientations.
        """

        diff = abs(theta1 - theta2)

        diff = min(diff, 180.0 - diff)

        return np.exp(
            -(diff ** 2) / (2.0 * 25.0 ** 2)
        )

    def area_similarity(a1, a2):

        """
        Compare contour areas using logarithmic ratio.
        """

        ratio = np.log(
            max(a1, a2) /
            max(1.0, min(a1, a2))
        )

        return np.exp(
            -(ratio ** 2) / (2.0 * 0.8 ** 2)
        )

    # =========================================================
    # Traffic Sign Color Ratio
    # =========================================================

    def _traffic_color_ratio(image, cnt):

        """
        Compute ratio of pixels inside contour
        that resemble traffic sign colors.
        """

        contour_mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8
        )

        cv2.drawContours(
            contour_mask,
            [cnt],
            -1,
            255,
            thickness=-1
        )

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        pixels = hsv[contour_mask == 255]

        if pixels.size == 0:
            return 0.0

        h = pixels[:, 0]
        s = pixels[:, 1]
        v = pixels[:, 2]

        # -----------------------------------------------------
        # Blue
        # -----------------------------------------------------

        blue_pixels = (
            (h >= 100) & (h <= 130) &
            (s >= 70) &
            (v >= 50)
        )

        # -----------------------------------------------------
        # Red
        # -----------------------------------------------------

        red_pixels = (
            (
                ((h >= 0) & (h <= 10)) |
                ((h >= 160) & (h <= 179))
            ) &
            (s >= 100) &
            (v >= 80)
        )

        # -----------------------------------------------------
        # Yellow
        # -----------------------------------------------------

        yellow_pixels = (
            (h >= 15) & (h <= 35) &
            (s >= 90) &
            (v >= 90)
        )

        # -----------------------------------------------------
        # Orange
        # -----------------------------------------------------

        orange_pixels = (
            (h >= 5) & (h <= 25) &
            (s >= 100) &
            (v >= 80)
        )

        # -----------------------------------------------------
        # White
        # -----------------------------------------------------

        white_pixels = (
            (s <= 50) &
            (v >= 180)
        )

        sign_pixels = (
            blue_pixels |
            red_pixels |
            yellow_pixels |
            orange_pixels |
            white_pixels
        )

        return (
            float(np.count_nonzero(sign_pixels))
            / len(pixels)
        )

    # =========================================================
    # Average HSV Inside Contour
    # =========================================================

    def _average_hsv_for_contour(image, cnt):

        """
        Compute average HSV color inside contour.

        Hue is circular, so circular mean is used.
        """

        contour_mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8
        )

        cv2.drawContours(
            contour_mask,
            [cnt],
            -1,
            255,
            thickness=-1
        )

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        pixels = hsv[contour_mask == 255]

        if pixels.size == 0:
            return np.array(
                [0.0, 0.0, 0.0],
                dtype=np.float32
            )

        h = pixels[:, 0].astype(np.float32)
        s = pixels[:, 1].astype(np.float32)
        v = pixels[:, 2].astype(np.float32)

        # circular hue averaging

        angles = h * 2.0 * np.pi / 180.0

        mean_sin = np.mean(np.sin(angles))
        mean_cos = np.mean(np.cos(angles))

        mean_h = (
            np.degrees(
                np.arctan2(mean_sin, mean_cos)
            ) / 2.0
        ) % 180.0

        return np.array([
            mean_h,
            np.mean(s),
            np.mean(v)
        ], dtype=np.float32)

    # =========================================================
    # Hue Distance
    # =========================================================

    def _hue_distance(h1, h2):

        """
        Circular hue distance.
        """

        diff = abs(h1 - h2)

        return min(diff, 180.0 - diff)

    # =========================================================
    # Bounding Box Distance
    # =========================================================

    def _bbox_distance(box1, box2):

        """
        Compute minimum distance between rectangles.

        Overlapping boxes -> distance = 0
        """

        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        left = max(x1, x2)
        right = min(x1 + w1, x2 + w2)

        top = max(y1, y2)
        bottom = min(y1 + h1, y2 + h2)

        # overlap case

        if right > left and bottom > top:
            return 0.0

        dx = max(left - right, 0)
        dy = max(top - bottom, 0)

        return np.hypot(dx, dy)

    # =========================================================
    # Contour Orientation
    # =========================================================

    def _contour_orientation(cnt):

        """
        Estimate dominant contour orientation using PCA.
        """

        pts = cnt.reshape(-1, 2).astype(np.float32)

        if pts.shape[0] < 5:
            return 0.0

        mean = pts.mean(axis=0)

        centered = pts - mean

        cov = np.cov(centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eig(cov)

        principal_vector = eigenvectors[
            :,
            np.argmax(eigenvalues)
        ]

        angle = np.degrees(
            np.arctan2(
                principal_vector[1],
                principal_vector[0]
            )
        )

        return angle

    # =========================================================
    # Merge Similar Contours
    # =========================================================

    def _merge_contours(contours, image):

        """
        Merge fragmented contours likely belonging
        to same traffic sign.
        """

        n = len(contours)

        if n <= 1:
            return contours

        features = []

        # -----------------------------------------------------
        # Extract contour features
        # -----------------------------------------------------

        for cnt in contours:

            area = cv2.contourArea(cnt)

            x, y, w, h = cv2.boundingRect(cnt)

            aspect_ratio = (
                w / float(h)
                if h > 0 else 0.0
            )

            hsv_avg = _average_hsv_for_contour(
                image,
                cnt
            )

            orientation = _contour_orientation(cnt)

            features.append({
                'contour': cnt,
                'area': area,
                'bbox': (x, y, w, h),
                'aspect_ratio': aspect_ratio,
                'hsv_avg': hsv_avg,
                'orientation': orientation
            })

        # -----------------------------------------------------
        # Union-Find
        # -----------------------------------------------------

        parent = list(range(n))

        def find(i):

            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]

            return i

        def union(i, j):

            ri = find(i)
            rj = find(j)

            if ri != rj:
                parent[rj] = ri

        # -----------------------------------------------------
        # Weights
        # -----------------------------------------------------

        W_DISTANCE = 0.25
        W_COLOR = 0.35
        W_ASPECT = 0.10
        W_ORIENTATION = 0.15
        W_AREA = 0.15

        DISTANCE_SIGMA = 50
        COLOR_SIGMA = 35

        MERGE_THRESHOLD = 0.7

        # -----------------------------------------------------
        # Pairwise affinity
        # -----------------------------------------------------

        for i in range(n):

            for j in range(i + 1, n):

                # spatial similarity

                distance = _bbox_distance(
                    features[i]['bbox'],
                    features[j]['bbox']
                )

                spatial_score = gaussian_similarity(
                    distance,
                    DISTANCE_SIGMA
                )

                # color similarity

                hsv_i = features[i]['hsv_avg']
                hsv_j = features[j]['hsv_avg']

                dh = _hue_distance(
                    hsv_i[0],
                    hsv_j[0]
                )

                ds = abs(hsv_i[1] - hsv_j[1])

                dv = abs(hsv_i[2] - hsv_j[2])

                color_distance = np.sqrt(
                    dh ** 2 +
                    ds ** 2 +
                    dv ** 2
                )

                color_score = gaussian_similarity(
                    color_distance,
                    COLOR_SIGMA
                )

                # aspect ratio similarity

                aspect_score = aspect_similarity(
                    features[i]['aspect_ratio'],
                    features[j]['aspect_ratio']
                )

                # orientation similarity

                orientation_score = (
                    orientation_similarity(
                        features[i]['orientation'],
                        features[j]['orientation']
                    )
                )

                # area similarity

                area_score = area_similarity(
                    features[i]['area'],
                    features[j]['area']
                )

                # combined affinity

                affinity = (
                    W_DISTANCE * spatial_score +
                    W_COLOR * color_score +
                    W_ASPECT * aspect_score +
                    W_ORIENTATION * orientation_score +
                    W_AREA * area_score
                )

                # merge contours

                if affinity >= MERGE_THRESHOLD:

                    union(i, j)

        # -----------------------------------------------------
        # Build contour groups
        # -----------------------------------------------------

        groups = {}

        for i in range(n):

            root = find(i)

            groups.setdefault(root, []).append(i)

        # -----------------------------------------------------
        # Merge grouped contours
        # -----------------------------------------------------

        merged = []

        for indices in groups.values():

            # single contour

            if len(indices) == 1:

                merged.append(
                    features[indices[0]]['contour']
                )

                continue

            # merge cluster

            combined = np.vstack([
                features[i]['contour']
                for i in indices
            ])

            hull = cv2.convexHull(combined)

            merged.append(hull)

        return merged

    # =========================================================
    # Extract Contours
    # =========================================================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # =========================================================
    # Merge Fragmented Contours
    # =========================================================

    contours = _merge_contours(
        contours,
        roi_image
    )

    # =========================================================
    # Detection Loop
    # =========================================================

    output = roi_image.copy()

    detected = 0

    roi_h, roi_w = roi_image.shape[:2]

    for cnt in contours:

        # -----------------------------------------------------
        # Basic geometry
        # -----------------------------------------------------

        area = cv2.contourArea(cnt)

        if area <= 0:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if h <= 0:
            continue

        aspect_ratio = w / float(h)

        rect_area = w * h

        fill_ratio = (
            area / rect_area
            if rect_area > 0 else 0.0
        )

        # solidity

        hull = cv2.convexHull(cnt)

        hull_area = cv2.contourArea(hull)

        solidity = (
            area / hull_area
            if hull_area > 0 else 0.0
        )

        # -----------------------------------------------------
        # Position filtering
        # -----------------------------------------------------

        center_x = x + w // 2
        center_y = y + h // 2

        if (
            center_y > roi_h * 0.92 or
            center_y < roi_h * 0.03
        ):
            continue

        if (
            center_x < roi_w * 0.08 or
            center_x > roi_w * 0.95
        ):
            continue

        # -----------------------------------------------------
        # Shape features
        # -----------------------------------------------------

        perimeter = cv2.arcLength(
            cnt,
            True
        )

        circularity = (
            4 * np.pi * area /
            (perimeter * perimeter)
            if perimeter > 0 else 0.0
        )

        approx = cv2.approxPolyDP(
            cnt,
            0.02 * perimeter,
            True
        )

        vertices = len(approx)

        # -----------------------------------------------------
        # Color consistency
        # -----------------------------------------------------

        color_ratio = _traffic_color_ratio(
            roi_image,
            cnt
        )

        # -----------------------------------------------------
        # Filtering
        # -----------------------------------------------------

        if aspect_ratio < 0.25 or aspect_ratio > 3.5:
            continue

        if fill_ratio < 0.18:
            continue

        if solidity < 0.45:
            continue

        if color_ratio < 0.40:
            continue

        # -----------------------------------------------------
        # Shape semantic reasoning
        # -----------------------------------------------------

        shape_ok = False

        # triangle sign

        if vertices == 3:

            shape_ok = (
                0.4 <= aspect_ratio <= 1.6 and
                fill_ratio > 0.20
            )

        # rectangle sign

        elif vertices == 4:

            shape_ok = (
                fill_ratio > 0.22 and
                0.3 <= aspect_ratio <= 3.2
            )

        # polygon / circular sign

        elif 5 <= vertices <= 10:

            shape_ok = (
                circularity > 0.38 and
                fill_ratio > 0.22
            ) or vertices == 8

        # highly circular

        elif circularity > 0.65:

            shape_ok = True

        if not shape_ok:
            continue

        # -----------------------------------------------------
        # Detection accepted
        # -----------------------------------------------------

        detected += 1

        # draw contour

        cv2.drawContours(
            output,
            [cnt],
            -1,
            (0, 255, 0),
            2
        )

        # draw bbox

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    return output, detected