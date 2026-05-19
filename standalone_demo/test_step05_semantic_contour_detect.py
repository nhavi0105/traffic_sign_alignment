"""STEP 4: Semantic Contour Detection
Input:
- Clean filtered mask
- ROI image

Process:
1. Contour extraction (biến white blobs từ step 3 thành individual objects)
2. Apply contour filtering (loại bớt những contours ít có khả năng là traffic signs) 
A. Area Filter (loại vì traffic signs thường không quá nhỏ)
- remove tiny contours 

B. Position Filter (loại vì traffic signs thường không nằm sát đáy ảnh)
- remove contours too low
- remove contours too far left

C. Aspect Ratio Filter (loại vì traffic signs thường không quá mỏng)
- remove thin vertical objects

3. Draw:
- Green contours (shape thật của object) (cho biết segmentation có chính xác không)
- Blue bounding boxes (hình chữ nhật nhỏ nhất chứa object) (cho biết object location)

Output:
- Final traffic sign detection result 
"""


import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# ===================================================
# STEP 5 — SEMANTIC CONTOUR DETECTION
# Pipeline:
# STEP 1 → ROI
# STEP 2 → HSV filtering
# STEP 3 → Morphology + Area filtering
# STEP 4 → Contour extraction
# STEP 5 → Semantic contour detection
# ===================================================

# ===================================================
# STEP 1 — SIMPLE ROI
# ===================================================   


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


# ===================================================
# STEP 2 — HSV FILTERING
# ===================================================

def hsv_filter(roi):

    blurred = cv2.GaussianBlur(
        roi,
        (5, 5),
        0
    )

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


# ===================================================
# STEP 3 — MORPHOLOGY
# ===================================================

def clean_mask(mask):

    # ------------------------------------------------
    # Morphological Closing
    # Fill small holes inside signs
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
    # AREA FILTERING
    # ------------------------------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed,
        connectivity=8
    )

    filtered = np.zeros_like(closed)

    MIN_AREA = 120

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area > MIN_AREA:

            filtered[labels == i] = 255

    return filtered


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

def semantic_contour_detection(mask, roi_image):
    """
    Semantic contour detection pipeline for traffic signs.
    
    Pipeline:
    1. Detect blue and red regions in the image
    2. Apply morphological operations to clean the masks
    3. Extract contours from the cleaned masks
    4. Filter contours based on area, position, and aspect ratio
    5. Perform semantic analysis on the remaining contours

    Args:
        mask (_type_): _description_
        roi_image (_type_): _description_
    Returns:
        detected_contours (list): List of contours that are mostlikely to be traffic signs after semantic analysis
    """
    # ADD erosion to separate connected signs, this can help to improve the contour extraction step by breaking apart signs that may be touching or very close to each other in the mask. By applying erosion before contour extraction, we can create a small gap between connected signs, which allows the contour extraction algorithm to identify them as separate contours. This can lead to more accurate detection of individual signs in cases where they are clustered together in the image.
    #kernel = cv2.getStructuringElement(
    #  cv2.MORPH_ELLIPSE,
    #  (3, 3)
    #)
    #mask = cv2.erode(mask, kernel, iterations=1) # apply erosion to the mask to separate connected signs before contour extraction

   

    def _traffic_color_ratio(image, cnt):

        contour_mask = np.zeros(image.shape[:2], dtype=np.uint8) # tạo mask đen cùng kích thước với ảnh gốc
        cv2.drawContours(contour_mask, [cnt], -1, 255, thickness=-1) # vẽ contour lên mask với màu trắng (255) và fill (-1) để có mask chỉ chứa contour đó
       
        # plot contour mask để debug
        plt.imshow(contour_mask, cmap='gray')
        plt.title('Contour Mask')
        plt.axis('off')
        plt.show()
        
        # tai sao contour mask lai bao gom nhieu contour khac nhau? contour mask chi nen co 1 contour, neu co nhieu contour thi co the do contour extraction tu step 4 van con nhieu contour thua

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # chuyển ROI sang HSV để phân tích màu sắc bên trong contour
        pixels = hsv[contour_mask == 255] # lấy tất cả pixel trong ROI mà nằm trong contour (nơi contour_mask là trắng) để phân tích màu sắc của chúng

        if pixels.size == 0:
            return 0.0

        h = pixels[:, 0]
        s = pixels[:, 1]
        v = pixels[:, 2]

        blue_pixels = (
            (h >= 100) & (h <= 130) & # blue hue upper range 
            (s >= 70) & (s <= 255) & # maximum saturation to avoid very grayish pixels
            (v >= 50) & (v <= 255) # minimum brightness to avoid very dark pixels
        )

        red_pixels = (
            (((h <= 10) & (h >= 0)) | ((h >= 160) & (h <= 179))) &
            (s >= 100) & (s <= 255) &
            (v >= 80) & (v <= 255)
        )

        yellow_pixels = (
            (h >= 15) & (h <= 35) &
            (s >= 90) & (v >= 90)
        )

        orange_pixels = (
            (h >= 5) & (h <= 25) &
            (s >= 100) & (v >= 80)
        )

        white_pixels = (
            (s <= 50) &
            (v >= 180)
        )

        sign_pixels = blue_pixels | red_pixels | yellow_pixels | orange_pixels | white_pixels
        return float(np.count_nonzero(sign_pixels)) / len(pixels) # ratio of sign-colored pixels to total contour pixels

    # =========================================================
    # Average HSV Inside Contour
    # =========================================================
    
    def _average_hsv_for_contour(image, cnt):
        # compute the average HSV color inside a contour region
        # traffic signs often have a dominant color, so calculating the average HSV can help determine if the contour has color characteristics consistent with traffic signs. This can be used as an additional criterion for filtering contours in the semantic contour detection step.
        
        # create a mask for the contour
        contour_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], -1, 255, thickness=-1) # fill the contour area with white (255) on the mask
        
        # convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # extract the HSV values of the pixels inside the contour
        pixels = hsv[contour_mask == 255]
        
        # handle empty contour case
        if pixels.size == 0:
            return (0, 0, 0)
        
        # hue is circular, so we need to compute the average hue carefully
        h = pixels[:, 0]
        s = pixels[:, 1]
        v = pixels[:, 2]    
        
        # convert hue to circular angles 
        angles = h * 2 * np.pi / 180.0
        
        # circular mean computation
        mean_sin = np.mean(np.sin(angles))
        mean_cos = np.mean(np.cos(angles))
        
        # compute average hue from mean sine and cosine
        average_hue = np.arctan2(mean_sin, mean_cos) * 180.0 / np.pi / 2.0
        
        # return average hsv vector
        return np.array([average_hue, np.mean(s), np.mean(v)], dtype=np.float32)
        
    # =========================================================
    # Hue Distance
    # =========================================================
    def _hue_distance(hue1, hue2):
        # compute the distance between two hue values, taking into account the circular nature of hue
        # this can be used to compare the average hue of a contour with the expected hue ranges for traffic signs to determine if the contour has a color consistent with traffic signs. A small hue distance would indicate that the contour's average color is close to the expected traffic sign colors, which can be a positive indicator for keeping that contour in the semantic contour detection step.
        
        diff = abs(hue1 - hue2) % 180
        return min(diff, 180 - diff)
    
    # =========================================================
    # Bounding Box Distance
    # =========================================================
    def __bbox_distance(box1, box2):
        # compute the distance between two bounding boxes, which can be used to determine if two contours are close enough to be considered part of the same sign. If the distance between their bounding boxes is below a certain threshold, we can consider merging them together in the semantic contour detection step to create a more complete contour for the sign.
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # compute overlap area
        left = max(x1, x2) 
        right = min(x1 + w1, x2 + w2)
        top = max(y1, y2)
        bottom = min(y1 + h1, y2 + h2)
        
        # boxes overlap if right > left and bottom > top
        if right > left and bottom > top: # if the boxes overlap, we can consider them as part of the same sign, so we return a distance of 0 to indicate that they are close enough to be merged together.
            return 0.0
        
        # horizontal gap
        dx = max(left - right, 0)
        
        # vertical gap
        dy = max(top - bottom, 0)
        
        # eucludean distance
        
        return np.hypot(dx, dy)
    
    # compute center point of a bounding box for spatial analysis, this can be used to analyze the spatial distribution of contours and determine if they are likely to belong to the same sign based on their proximity to each other. For example, if two contours have center points that are close to each other, we can consider them as part of the same sign and potentially merge them together in the semantic contour detection step to create a more complete contour for the sign.
    
    
    # =========================================================
    # Contour Center
    # =========================================================
    def _contour_center(box):
        
        x, y, w, h = box
        
        return np.array([x + w / 2, y + h / 2], dtype=np.float32)
    
    
    
    # =========================================================
    # Contour Orientation
    # estimate contour orientation using PCA, this can be used to analyze the alignment of contours and determine if they are likely to belong to the same sign based on their orientation. For example, if two contours have similar orientations, we can consider them as part of the same sign and potentially merge them together in the semantic contour detection step to create a more complete contour for the sign.
    # Why:
    # Fragments belonging to the same sign
    # often share similar orientation.
    #
    # PCA finds:
    # - major axis direction
    # - dominant geometric orientation
    #
    # Example:
    # elongated contour tilted 30 degrees
    # -> orientation ≈ 30
    #
    # Returns:
    # angle in degrees
    # =========================================================
    def _contour_orientation(cnt):
        # flatten contour points 
        pts = cnt.reshape(-1, 2).astype(np.float32)
        
        # PCA becomes unstable with too few points 
        if pts.shape[0] < 5:
            return 0.0
        
        # compute contour centroid 
        mean = pts.mean(axis=0)
        
        # center points around origin
        centered = pts - mean
        
        # compute covariance matrix
        cov = np.cov(centered, rowvar=False)
        
        # eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # principal direction is eigenvector with largest eigenvalue
        principal_vector = eigenvectors[:, np.argmax(eigenvalues)]
        
        # compute angle of principal vector
        angle = np.arctan2(principal_vector[1], principal_vector[0]) * 180.0 / np.pi
        
        return angle
    
   
    # =========================================================
    # Similarity Functions
    # =========================================================
    
    def gaussian_similarity(distance, sigma):
        # convert a distance value into a similarity score using a Gaussian function, where smaller distances will yield similarity scores closer to 1, and larger distances will yield similarity scores closer to 0. This can be used to compute similarity scores for spatial proximity, color similarity, aspect ratio similarity, and orientation similarity between contours in the semantic contour detection step, which can then be combined to determine if contours should be merged together.
        return np.exp(- (distance ** 2) / (2 * sigma ** 2))
    
    def aspect_similarity(aspect1, aspect2):
        
        if aspect1 == 0 or aspect2 == 0:
            return 0.0
        # compute the similarity between two aspect ratios, this can be used to determine if two contours have similar shapes and are likely to belong to the same sign. For example, if two contours have aspect ratios that are close to each other, we can consider them as part of the same sign and potentially merge them together in the semantic contour detection step to create a more complete contour for the sign.
        ratio = abs(aspect1 - aspect2) / max(aspect1, aspect2)
        
        return np.exp(-(ratio ** 2) / (2 * 0.6 ** 2)) # 0.6 is the aspect ratio similarity threshold, contours with aspect ratios that differ by less than 60% will have a similarity score above 0.5, which can be used as part of the criteria for merging contours together in the semantic contour detection step.

    def orientation_similarity(orientation1, orientation2):
        
        diff = abs(orientation1 - orientation2) 
        diff = min(diff, 180 - diff)
        
        return np.exp(-(diff ** 2) / (2 * 30 ** 2)) # 30 degrees is the orientation similarity threshold, contours with orientations that differ by less than 30 degrees will have a similarity score above 0.5, which can be used as part of the criteria for merging contours together in the semantic contour detection step.
    
    def area_similarity(area1, area2):
        
        ratio = np.log(max(area1, area2) / max(1.0, min(area1, area2))) # compute the area similarity between two contours by taking the logarithm of the ratio of their areas. This can be used to determine if two contours have compatible sizes and are likely to belong to the same sign. For example, if two contours have areas that are within a reasonable range of each other (e.g., one contour is not significantly larger than the other), we can consider them as part of the same sign and potentially merge them together in the semantic contour detection step to create a more complete contour for the sign.
        
        return np.exp(-(ratio ** 2) / (2 * 0.8 ** 2)) # 0.8 is the area similarity threshold, contours with area ratios that differ by less than a factor of e^0.8 (approximately 2.23) will have a similarity score above 0.5, which can be used as part of the criteria for merging contours together in the semantic contour detection step. This helps to ensure that we only merge contours that have compatible sizes, which can improve the accuracy of the merged contours representing actual traffic signs.
    
    def shape_similarity(v1, v2):
        # Compute the similarity between two shape descriptors (e.g., number of vertices)
        diff = abs(v1 - v2)
        return np.exp(-(diff ** 2) / (2 * 2 ** 2)) # 2 is the shape similarity threshold

    # merge contours that are close to each other and have similar color characteristics, this can help to create more complete contours for traffic signs that were fragmented into smaller pieces during contour extraction. By merging contours that are likely to belong to the same sign based on their proximity and color similarity, we can improve the accuracy of the semantic contour detection step by ensuring that we analyze more complete contours that better represent the actual traffic signs in the images.
    # Core idea:
    # Traffic signs may fragment into multiple contours due to:
    # - glare
    # - blur
    # - reflections
    # - partial occlusion
    # - broken edges
    #
    # Instead of treating contours independently,
    # we group contours using:
    #
    # - spatial proximity
    # - color similarity
    # - aspect ratio similarity
    # - orientation similarity
    # - area consistency
    #
    # Then contours in the same cluster
    # are merged into a single contour candidate.
   
    def _merge_contours(contours, image):
          
        # AREA
        W_AREA = 0.1
        
        # spatial threshold 
        DISTANCE_THRESHOLD = 50
        W_DISTANCE = 0.2
        
        # color similarity threshold in HSV space
        COLOR_DISTANCE_THRESHOLD = 35
        W_COLOR = 0.2
        
        # aspect ratio similarity threshold
        ASPECT_RATIO_THRESHOLD = 0.6
        W_ASPECT = 0.1
        
        # orientation similarity threshold
        ORIENTATION_THRESHOLD = 30
        W_ORIENTATION = 0.2
        
        # SHAPE
        W_SHAPE = 0.2
       
        # gaussian scales
        DISTANCE_SIGMA = 50
        COLOR_SIGMA = 35
        
        # merge probability threshold (combined similarity score)
        MERGE_THRESHOLD = 0.85
       
       
        
        n = len(contours)
        
        if n <= 1:
            return contours
        
        # store geometric and appearance features 
        features = []
        
        for cnt in contours:
            # contour area
            area = cv2.contourArea(cnt) # compute the area of the contour, which can be used as a feature for clustering contours together. Contours that belong to the same sign are likely to have compatible areas, so we can use area as one of the criteria for determining if contours should be merged together in the semantic contour detection step.
            
            # bounding box
            x, y, w, h = cv2.boundingRect(cnt) # bounding box of the contour, represented as (x, y, w, h) where (x, y) is the top-left corner and (w, h) are the width and height. This can be used to compute spatial relationships between contours and determine if they are close enough to be considered part of the same sign for potential merging in the semantic contour detection step.
            
            # aspect ratio
            aspect_ratio = w / float(h) if h > 0 else 0
            
            # average hsv color
            hsv_avg = _average_hsv_for_contour(image, cnt)
            
            # dominant orientation
            orientation = _contour_orientation(cnt)
            
            # shape descriptors
            perimeter = cv2.arcLength(cnt, True) # compute the perimeter of the contour, which can be used as a shape descriptor to help determine if contours are likely to belong to the same sign. Contours that belong to the same sign are likely to have similar shape characteristics, so we can use perimeter as one of the features for clustering contours together in the semantic contour detection step.
            approx = cv2.approxPolyDP(
                cnt, 0.02 * perimeter, True) # approximate the contour to a simpler shape using the Ramer-Douglas-Peucker algorithm, which can help to reduce noise and create a more stable representation of the contour's shape. This can be useful for analyzing the geometric characteristics of the contour and determining if it is likely to belong to a traffic sign based on its shape features in the semantic contour detection step.
            vertices = len(approx) # number of vertices in the approximated contour, which can be used as a shape descriptor to help determine if contours are likely to belong to the same sign. For example, certain types of traffic signs may have characteristic numbers of vertices (e.g., octagonal stop signs), so this feature can be useful for clustering contours together in the semantic contour detection step based on their shape characteristics.
            
            
            # store features for clustering
            features.append({
                'contour': cnt,
                'area': area,
                'bbox': (x, y, w, h),
                'aspect_ratio': aspect_ratio,
                'hsv_avg': hsv_avg,
                'orientation': orientation,
                'vertices': vertices
            })
           
        # union-find structure for contours clustering
        parent = list(range(n)) # initially each contour is its own cluster
        
        # find cluster root
        def find(i):
            # find root of cluster containing contour i with path compression
            # for example, if contour 0 and contour 2 are in the same cluster, then find(0) and find(2) will return the same root index, indicating that they belong to the same cluster. This allows us to group contours together based on their similarity and proximity, which can help to create more complete contours for traffic signs in the semantic contour detection step.
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        
        def union(i, j):
            # union clusters containing contour i and contour j
            # for example, if contour 0 and contour 2 are determined to be similar and close enough to be merged together, we can call union(0, 2) to indicate that they belong to the same cluster. This way, when we later check the clusters of contours, we will know that contour 0 and contour 2 should be treated as part of the same sign and potentially merged together in the semantic contour detection step.
            root_i = find(i)
            root_j = find(j)
            
            if root_i != root_j:
                parent[root_j] = root_i
      
        
        # compare every contour pair 
        for i in range(n):
            for j in range(i + 1, n):
                
                # bounding boxes
                box_i = features[i]['bbox']  # accessing the bounding box of the i-th contour from the features list, which was computed earlier when we extracted features for each contour. The bounding box is represented as a tuple (x, y, w, h) where (x, y) is the top-left corner of the bounding box and (w, h) are its width and height. We will use these bounding boxes to compute spatial distance between contours and determine if they are close enough to be considered part of the same sign for potential merging in the semantic contour detection step.
                box_j = features[j]['bbox']  
                
                # spatial distance 
                distance = __bbox_distance(box_i, box_j)
                
                spatial_score = gaussian_similarity(distance, DISTANCE_SIGMA) # convert the spatial distance into a similarity score using a Gaussian function, where contours that are close to each other will have a similarity score close to 1, and contours that are far apart will have a similarity score close to 0. This spatial similarity score can be used as part of the combined criteria for determining if two contours should be merged together in the semantic contour detection step.
                
                # average hsv colors 
                hsv_i = features[i]['hsv_avg']
                hsv_j = features[j]['hsv_avg']
                
                # circular hue distance
                dh = _hue_distance(hsv_i[0], hsv_j[0]) # compute the hue distance between the average HSV colors of contour i and contour j using the _hue_distance function, which takes into account the circular nature of hue. This hue distance can be used as part of the criteria for determining if two contours are similar enough in color to be considered part of the same sign for potential merging in the semantic contour detection step. A small hue distance would indicate that the contours have similar color characteristics, which can be a positive indicator for merging them together.
                
                ds = abs(hsv_i[1] - hsv_j[1]) # compute the saturation distance between the average HSV colors of contour i and contour j, which can also be used as part of the criteria for determining if two contours are similar enough in color to be considered part of the same sign for potential merging in the semantic contour detection step. A small saturation distance would indicate that the contours have similar color characteristics, which can be a positive indicator for merging them together.
                dv = abs(hsv_i[2] - hsv_j[2]) # compute the value (brightness) distance between the average HSV colors of contour i and contour j, which can also be used as part of the criteria for determining if two contours are similar enough in color to be considered part of the same sign for potential merging in the semantic contour detection step. A small value distance would indicate that the contours have similar color characteristics, which can be a positive indicator for merging them together.
                
                color_distance = np.sqrt(dh * dh + ds * ds + dv * dv) # compute the overall color distance between contour i and contour j by combining the hue distance, saturation distance, and value distance using a Euclidean distance formula. This overall color distance can be used as part of the criteria for determining if two contours are similar enough in color to be considered part of the same sign for potential merging in the semantic contour detection step. A small overall color distance would indicate that the contours have similar color characteristics, which can be a positive indicator for merging them together.
                
                color_score = gaussian_similarity(color_distance, COLOR_SIGMA) # convert the color distance into a similarity score using a Gaussian function, where contours with a color distance close to 0 will have a similarity score close to 1, and contours with a color distance much larger than the COLOR_DISTANCE_THRESHOLD will have a similarity score close to 0. This color similarity score can be used as part of the combined criteria for determining if two contours should be merged together in the semantic contour detection step.

                # aspect ratio similarity
                aspect_i = features[i]['aspect_ratio']
                aspect_j = features[j]['aspect_ratio']
                
                aspect_score = aspect_similarity(aspect_i, aspect_j) # compute the aspect ratio similarity score between contour i and contour j using an aspect_similarity function that compares their aspect ratios and returns a similarity score based on how close they are to each other. This aspect ratio similarity score can be used as part of the combined criteria for determining if two contours should be merged together in the semantic contour detection step, as contours that belong to the same sign are likely to have similar aspect ratios.

                # orientation similarity
                orientation_i = features[i]['orientation']
                orientation_j = features[j]['orientation']
                
                orientation_score = orientation_similarity(orientation_i, orientation_j) # compute the orientation similarity score between contour i and contour j using an orientation_similarity function that compares their orientations and returns a similarity score based on how close they are to each other. This orientation similarity score can be used as part of the combined criteria for determining if two contours should be merged together in the semantic contour detection step, as contours that belong to the same sign are likely to have similar orientations.
                
                # area consistency score, same sign contours are likely to have compatible areas
                area_i = features[i]['area']
                area_j = features[j]['area']
                area_score = area_similarity(area_i, area_j) # compute the area consistency score between contour i and contour j using an area_similarity function that compares their areas and returns a similarity score based on how close they are to each other. This area consistency score can be used as part of the combined criteria for determining if two contours should be merged together in the semantic contour detection step, as contours that belong to the same sign are likely to have compatible areas.
                
                # shape similarity score, same sign contours are likely to have similar shape characteristics (e.g., number of vertices)
                vertices_i = features[i]['vertices']
                vertices_j = features[j]['vertices']
                shape_score = shape_similarity(vertices_i, vertices_j) # compute the shape similarity score between
                
                affinity = (W_DISTANCE * spatial_score +
                            W_COLOR * color_score +
                            W_ASPECT * aspect_score +
                            W_ORIENTATION * orientation_score +
                            W_AREA * area_score +
                            W_SHAPE * shape_score) # compute a combined affinity score between contour i and contour j by weighting the spatial similarity score, color similarity score, aspect ratio similarity score, orientation similarity score, area consistency score, and shape similarity score using predefined weights (W_DISTANCE, W_COLOR, W_ASPECT, W_ORIENTATION, W_AREA, W_SHAPE). This combined affinity score can be used to determine if the two contours should be merged together in the semantic contour detection step. If the affinity score is above a certain threshold (MERGE_THRESHOLD), we can consider the contours as belonging to the same sign and merge them together.

                if affinity > MERGE_THRESHOLD:
                    union(i, j) # if the combined affinity score between contour i and contour j is greater than the MERGE_THRESHOLD, we call the union function to indicate that they belong to the same cluster. This means that we will treat these two contours as part of the same sign and potentially merge them together in the semantic contour detection step to create a more complete contour for the sign.
       
        # build contours groups based on union-find structure
        # for example
        groups = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)
            
        # merge contours in the same cluster
        merged_contours = []
        
        for indices in groups.values(): # iterate through each cluster of contours that belong together
            
            # single contour cluster 
            if len(indices) == 1:
                merged_contours.append(features[indices[0]]['contour']) # if the cluster contains only one contour, we can simply add that contour to the merged_contours list without any merging, as it is already a complete contour for analysis in the semantic contour detection step.
                continue
            
            # combine contours in the same cluster into a single contour
            combined = np.vstack([features[i]['contour'] for i in indices]) # if the cluster contains multiple contours, we can combine them into a single contour by vertically stacking their contour points together. This creates a new contour that represents the merged shape of all the contours in the cluster, which can then be analyzed in the semantic contour detection step to determine if it is a valid traffic sign contour.
            
            # compute convex hull of combined contour to create a single unified contour that encompasses all the contours in the cluster, which can help to create a more complete contour for the sign and improve the accuracy of the semantic contour detection step by ensuring that we analyze contours that better represent the actual traffic signs in the images.
            hull = cv2.convexHull(combined)
            
            merged_contours.append(hull) # add the merged contour (convex hull) to the list of merged_contours, which will be used for further analysis in the semantic contour detection step to determine if it is a valid traffic sign contour.
        
        return merged_contours # return the list of merged contours that represent potential traffic signs after combining contours that are likely to belong to the same sign based on their spatial proximity, color similarity, aspect ratio similarity, orientation similarity, and area consistency. These merged contours can then be analyzed in the semantic contour detection step to determine if they are valid traffic sign contours.
    

    contours, _ = cv2.findContours( # tìm contours trên mask đã được làm sạch và lọc màu từ step 3

        mask,

        cv2.RETR_EXTERNAL, # chỉ lấy contours ngoài cùng (external contours) để tránh bị nhiễu bởi các contour lồng nhau

        cv2.CHAIN_APPROX_SIMPLE # giảm số điểm trong contour để đơn giản hóa hình dạng, giúp loại bỏ những chi tiết nhỏ không cần thiết

    )


   
    # call merge_contours here to combine contours that likely belong to the same sign before applying semantic checks, this way we can create more complete contours for analysis and improve the chances of correctly identifying valid traffic signs. By merging contours that are close to each other and have similar color characteristics, we can reduce fragmentation and ensure that we analyze contours that better represent the actual traffic signs in the images, which can lead to more accurate detection results in the semantic contour detection step.
    
    # steps to merge small contours that belong to the same sign before applying semantic checks, this way we can avoid losing valid signs that were fragmented into smaller contours during extraction. We can use criteria such as proximity (contours that are close to each other), color similarity (contours that have similar color characteristics), and shape similarity (contours that have similar aspect ratio or orientation) to determine if small contours should be merged together. By merging small contours that likely belong to the same sign, we can create a more complete contour for the sign, which can then be analyzed with the semantic checks to determine if it is a valid traffic sign. This approach helps to improve the accuracy of the detection by reducing fragmentation and ensuring that valid signs are not discarded due to being split into multiple small contours.
    # we can also consider using a more advanced technique such as contour clustering or hierarchical contour analysis to group contours that are likely to belong to the same sign based on their spatial relationships and characteristics. This way, we can create a more robust detection pipeline that can handle cases where signs are partially obscured, damaged, or have complex shapes that result in multiple contours being extracted. By effectively merging contours that belong to the same sign, we can improve the overall performance of the semantic contour detection step and increase the chances of correctly identifying valid traffic signs in the images.
    
    
    # spatial distance threshold for merging contours, signs that belong to the same sign are likely to be close to each other, we can set a distance threshold (e.g., 50 pixels) and merge contours that are within this distance from each other. This helps to combine fragmented contours that likely belong to the same sign into a single contour for analysis.
    
    # color similarity threshold for merging contours, signs that belong to the same sign are likely to have similar color characteristics, we can calculate the average color of each contour and merge contours that have a color distance below a certain threshold (e.g., using Euclidean distance in HSV color space). This helps to ensure that we only merge contours that are likely to belong to the same sign based on their color properties.
    
    # alignment similarity: traffic sign fragmentsoften align circularly or along a certain orientation, we can analyze the orientation of each contour (e.g., using PCA to find the major axis) and merge contours that have similar orientations and are spatially close to each other. This helps to combine contours that are likely part of the same sign based on their alignment and spatial relationship.
    
    # scale consistency: fragments belong to same sign tend to have compatible sizes 
    
    # shape compatibility: arcs belong to circle, rectangles belong to rectangle sign
   
    # =========================================================
    # Merge Fragmented Contours
    # =========================================================
    
    #print number of contours before merging, this can help us to understand how many fragmented contours we are starting with before applying the merging process. By printing the number of contours before merging, we can get a sense of the level of fragmentation in the contour extraction step and how much improvement we might expect from the merging process in the semantic contour detection step.
    print(f"Number of contours before merging: {len(contours)}")
    
    contours = _merge_contours(
        contours,
        roi_image
    )
    
    # print number of contours after merging, this can help us to understand how many potential sign contours we are analyzing in the semantic contour detection step after combining fragmented contours together. By printing the number of contours after merging, we can get a sense of how effective the merging process is at reducing fragmentation and creating more complete contours for analysis in the next step.
    print(f"Number of contours after merging: {len(contours)}")
    
    
    # =========================================================
    # Detection Loop with Semantic Checks on combined contours
    # =========================================================
    output = roi_image.copy() # create a copy of the ROI image to draw detected contours on it for visualization purposes, this way we can see which contours were detected as valid traffic signs after applying the semantic checks. By drawing the detected contours on a copy of the ROI image, we can visually verify the results of the semantic contour detection step and ensure that the contours we identified as valid traffic signs are correctly highlighted in the output image.
    detected = 0
    roi_h, roi_w = roi_image.shape[:2] # lấy chiều cao và chiều rộng của ROI để sử dụng trong các bước lọc tiếp theo, ví dụ: để xác định vị trí tương đối của contour trong ROI hoặc để tính toán tỷ lệ diện tích của contour so với kích thước ROI
    # we should take the roi of the bounding box for the relative area check, if the contour is too small relative to the roi size then we can discard it as noise, if the contour is too large relative to the roi size then we can discard it as unlikely sign or non-sign object. This helps to filter out contours that are not likely to be valid traffic signs based on their size relative to the ROI, which can improve the accuracy of the detection by reducing false positives from contours that are too small (e.g., due to noise) or too large (e.g., due to non-sign objects).
    
    # plot combined contours for visualization, this can help us to visually verify that the merging process is working correctly and that we are creating more complete contours for the signs. By plotting the combined contours on the ROI image, we can see how the contours have been merged together and ensure that they better represent the actual traffic signs in the images before applying the semantic checks.
    for cnt in contours:
        cv2.drawContours(output, [cnt], -1, (255, 0, 0), 2) # draw the combined contours on the output image in blue color with a thickness of 2 pixels for visualization purposes. This allows us to see the results of the contour merging process and verify that we are creating more complete contours for the signs before applying the semantic checks in the next step.
        
    for cnt in contours:
        # plot contour to verify its existence
        canvas = output.copy()

        cv2.drawContours(canvas, [cnt], -1, (0,255,0), 2)

        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.title("Check existence")
        plt.axis('off')
        plt.show()
        
        # the 2 circles exist here => they be filtered in later steps 
        # -----------------------------------------------------
        # Basic geometry
        # -----------------------------------------------------

        area = cv2.contourArea(cnt) #  diện tích contour, signs thường có diện tích không quá nhỏ, nhung vi dang check tung contour mot nen không thể loại bỏ contour nhỏ ngay từ đầu, nhưng sẽ loại bỏ trong bước này nếu area quá nhỏ (ví dụ: do nhiễu hoặc chi tiết không phải là sign) để giảm số lượng contour cần phân tích tiếp theo. Nếu area quá nhỏ, có thể coi đó là nhiễu và bỏ qua contour đó.
        # area of the bounding box of the contour, this can be used to compute the fill ratio (extent) of the contour, which is the ratio of the contour area to the area of its bounding box. Signs typically have a reasonable fill ratio, so if the fill ratio is too low, it may indicate that the contour is a fragment or a non-sign object with a lot of empty space, and we can discard it as unlikely to be a valid traffic sign. This helps to filter out contours that are not likely to be valid traffic signs based on their shape and how well they fill their bounding box, which can improve the accuracy of the detection by reducing false positives from contours that are not well-defined or do not have the typical shape characteristics of traffic signs.
        area_bounding_box = cv2.boundingRect(cnt) # bounding box of the contour, represented as (x, y, w, h) where (x, y) is the top-left corner and (w, h) are the width and height. This can be used to compute the aspect ratio of the contour, which is the ratio of its width to its height. Signs typically have a reasonable aspect ratio (e.g., not too thin or too wide), so if the aspect ratio is outside of a certain range, it may indicate that the contour is unlikely to be a valid traffic sign (e.g., it could be a pole or a large billboard), and we can discard it as unlikely to be a valid traffic sign. This helps to filter out contours that are not likely to be valid traffic signs based on their shape characteristics, which can improve the accuracy of the detection by reducing false positives from contours that do not have the typical shape characteristics of traffic signs.
        # vi du neu hai contour chi belong to a sign thi lam sao check area cua contour do? neu contour do co area nho thi co the coi la nhiễu, neu contour do co area lon thi co the coi la sign, tuy nhien neu contour do chi la mot phan cua sign thi co the bi loai bo, do do can check them cac tieu chi khac nhu position, aspect ratio, color ratio de dam bao khong loai bo contour thuoc ve sign
        relative_area = area_bounding_box[2] * area_bounding_box[3] / (roi_h * roi_w) # tính diện tích của bounding box của contour so với diện tích của ROI, nếu tỷ lệ này quá nhỏ thì có thể coi đó là nhiễu hoặc chi tiết không phải là sign, nếu tỷ lệ này quá lớn thì có thể coi đó là vật thể lớn không phải sign. Việc check relative area giúp loại bỏ những contour có kích thước không phù hợp với kích thước của ROI, từ đó tăng độ chính xác của việc phát hiện bằng cách giảm số lượng contour cần phân tích tiếp theo.
        
        # bounding box aspect ratio, signs thường có aspect ratio không quá nhỏ (rất mỏng) hoặc quá lớn (rất rộng), nếu aspect ratio quá nhỏ có thể là vật thể dài và mỏng như cột điện, nếu aspect ratio quá lớn có thể là vật thể rộng như biển quảng cáo. Việc check aspect ratio giúp loại bỏ những contour có hình dạng không phù hợp với hình dạng chung của traffic signs.s
        x,y,w,h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        aspect_ratio = w / float(h) if h > 0 else 0
        
        # extent (fill ratio) contour co fill box tot khong, which is similar to relative area 
        fill_ratio = area / (area_bounding_box[2] * area_bounding_box[3]) if area_bounding_box[2] * area_bounding_box[3] > 0 else 0
        
        # solidity, signs thường có solidity không quá thấp, nếu solidity quá thấp có thể là vật thể có nhiều lỗ hổng hoặc hình dạng không đặc, trong khi traffic signs thường có hình dạng đặc và ít lỗ hổng. Việc check solidity giúp loại bỏ những contour có hình dạng không đặc trưng cho traffic signs.
        # nhung neu contour do chi la mot phan cua sign thi co the co solidity thap
        hull_area = cv2.convexHull(cnt)
        solidity = area / cv2.contourArea(hull_area) if cv2.contourArea(hull_area) > 0 else 0
        
        # check contour compact color, signs thường có màu sắc tập trung và rõ ràng, nếu contour có tỷ lệ màu sắc phù hợp với màu sắc đặc trưng của traffic signs (ví dụ: đỏ, xanh dương, vàng) thì có khả năng cao là contour đó thuộc về một traffic sign. Việc check tỷ lệ màu sắc giúp loại bỏ những contour có màu sắc không phù hợp với traffic signs, từ đó tăng độ chính xác của việc phát hiện.        
        color_ratio = _traffic_color_ratio(roi_image, cnt) # tính tỷ lệ pixel có màu sắc phù hợp với traffic signs trong contour, nếu tỷ lệ này cao thì có khả năng contour đó thuộc về một traffic sign, nếu thấp thì có thể là nhiễu hoặc vật thể không phải sign. Việc check color ratio giúp tăng độ chính xác của việc phát hiện bằng cách loại bỏ những contour có màu sắc không phù hợp với traffic signs.
        # vi du nếu contour có color ratio thấp thì có thể coi đó là nhiễu hoặc vật thể không phải sign, nếu contour có color ratio cao thì có thể coi đó là contour thuộc về sign, tuy nhiên cũng cần check thêm các tiêu chí khác như area, aspect ratio, solidity để đảm bảo không loại bỏ contour thuộc về sign chỉ vì color ratio thấp (ví dụ: do ánh sáng kém hoặc màu sắc bị mờ) hoặc không giữ lại contour không phải sign chỉ vì color ratio cao (ví dụ: do vật thể khác có màu sắc tương tự sign). Do đó, việc check color ratio nên được kết hợp với các tiêu chí khác để đạt được kết quả tốt nhất.

        # -----------------------------------------------------
        # Position filtering
        # -----------------------------------------------------
        
        center_x = x + w // 2 # tính tọa độ x trung tâm
        center_y = y + h // 2 # toa do y trung tam 
        
        #0.85, 0.03
        if center_y > roi_h * 0.95 or center_y < roi_h * 0.01: # signs thường không nằm sát đáy ảnh và cũng không nằm quá cao
            continue
        
        #0.3, 0.9 can discard blobs on the road on 1st image but 0.2, 0.9 cannot and filter 1 contour in 2nd image(rm:2)
        #0.2 and 0.9 probably filter out the 2circle sign in 2nd image(YESS)
        #if center_x < roi_w * 0.15 or center_x > roi_w * 0.9: # signs thường không nằm sát bên trái và cũng không nằm sát bên phải
        #   continue       
       
        # -----------------------------------------------------
        # Shape features
        # ----------------------------------------------------- 
        # print area, aspect ratio, fill ratio, solidity, color ratio for debugging
        print(f"Contour area: {area}, aspect ratio: {aspect_ratio:.2f}, fill ratio: {fill_ratio:.2f}, solidity: {solidity:.2f}, color ratio: {color_ratio:.2f}")
        
        perimeter = cv2.arcLength(cnt, True) # chu vi contour
        
        circularity = (4 * np.pi * area / 
                       (perimeter * perimeter) 
                       if perimeter > 0 else 0.0) # circularity đo mức độ giống hình tròn của contour, signs thường có circularity không quá thấp, nếu circularity quá thấp có thể là vật thể có hình dạng phức tạp hoặc không đặc trưng cho traffic signs, trong khi traffic signs thường có hình dạng đơn giản và đặc trưng. Việc check circularity giúp loại bỏ những contour có hình dạng không phù hợp với traffic signs.
        
        approx = cv2.approxPolyDP(cnt,
                                  0.02 * perimeter, 
                                  True) # làm mịn contour để giảm số lượng điểm, signs thường có hình dạng đơn giản với số lượng cạnh nhất định (ví dụ: hình tròn有 nhiều cạnh, hình tam giác有 3 cạnh, hình vuông有 4 cạnh), việc check số lượng cạnh sau khi làm mịn giúp loại bỏ những contour有 hình dạng quá phức tạp hoặc không đặc trưng cho traffic signs.
        
        vertices = len(approx) # số lượng đỉnh sau khi làm mịn contour, signs thường số lượng đỉnh nhất định (ví dụ: hình tròn有 nhiều đỉnh, hình tam giác有 3 đỉnh, hình vuông有 4 đỉnh), việc check số lượng đỉnh giúp loại bỏ những contour有 hình dạng không phù hợp với traffic signs. Tuy nhiên, cũng cần lưu ý rằng một số traffic signs có thể bị che khuất hoặc bị hỏng nên có thể không có đủ số lượng đỉnh rõ ràng, do đó việc check số lượng đỉnh nên được kết hợp với các tiêu chí khác để đảm bảo không loại bỏ contour thuộc về sign chỉ vì số lượng đỉnh không phù hợp.

        # -----------------------------------------------------
        # Color consistency
        # -----------------------------------------------------
        
        # we want to check combined conditions here since if a sign has low circularity but good color ratio and aspect ratio, it might still be a valid sign, or if a sign has low color ratio but good shape characteristics, it might still be worth keeping for further analysis. By checking combined conditions, we can avoid being too strict on any single criterion and instead look for contours that have a reasonable combination of characteristics that are consistent with traffic signs. This way, we can increase the chances of keeping valid signs while still filtering out obvious noise.

        color_ratio = _traffic_color_ratio(roi_image, cnt)
        
        # -----------------------------------------------------
        # Filtering
        # -----------------------------------------------------
        if aspect_ratio < 0.3 or aspect_ratio > 3.5:
            continue
        if fill_ratio < 0.15:
            continue
        if solidity < 0.25:
            continue
        if color_ratio < 0.40:
            continue

        # -----------------------------------------------------
        # Shape semantic reasoning
        # -----------------------------------------------------
        shape_ok = False
        print(f"perimeter: {perimeter}, circularity: {circularity:.2f}, vertices:{vertices}")
        # triangle sign
        if vertices == 3:
            shape_ok = (0.3 <= aspect_ratio <= 1.6 and 
                        fill_ratio > 0.15)
        # rectangle sign
        elif vertices == 4:
            shape_ok = (fill_ratio > 0.22 and 
                        0.3 <= aspect_ratio <= 3.2)
            
        # circular sign
        elif vertices >= 10:
            shape_ok = (circularity >= 0.28 and 
                        fill_ratio > 0.22
            ) or vertices == 8
        
        elif circularity > 0.5 and fill_ratio > 0.25:
            shape_ok = True

        if not shape_ok:
            continue
        print(f"shape ok: {shape_ok}")
        # -----------------------------------------------------
        # Detection accepted
        # -----------------------------------------------------

        detected += 1

        # draw contour of the detected sign in green with 
        cv2.drawContours(

            output,

            [cnt],

            -1,

            (0, 255, 0),

            5

        )
        # draw bounding box of the detected sign in blue
        # why there is blue bounding box on some signs but there is no green contour? because the blue bounding box is drawn for all contours that were extracted after merging, while the green contour is only drawn for those contours that passed the semantic checks and were accepted as valid traffic signs. So if a contour was extracted and merged but did not pass the semantic checks, it would have a blue bounding box but no green contour drawn on it. This allows us to visually differentiate between contours that were considered as potential signs (blue bounding box) and those that were actually detected as valid signs after applying the semantic checks (green contour).
        cv2.rectangle(

            output,

            (x, y),

            (x + w, y + h),

            (255, 0, 0),

            3

        )
    
        # -----------------------------------------------------
        # STEP 5A — EXTRACT TRAFFIC SIGN ROI
        # -----------------------------------------------------

        # bounding box for contour
        x, y, w, h = cv2.boundingRect(cnt)

        # small padding around sign
        padding = 8

        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)

        x2 = min(x + w + padding, roi_w)
        y2 = min(y + h + padding, roi_h)

        # extract sign region
        sign_roi = roi_image[y1:y2, x1:x2]

        # skip invalid crops
        if sign_roi.size == 0:
            continue

        # -----------------------------------------------------
        # OPTIONAL — Resize extracted sign
        # Useful for classifier input later
        # -----------------------------------------------------

        extracted_sign = cv2.resize(
            sign_roi,
            (128, 128)
        )

        # -----------------------------------------------------
        # SAVE EXTRACTED SIGN
        # -----------------------------------------------------

        os.makedirs(
            'output_images/extracted_signs',
            exist_ok=True
        )

        sign_path = (
            f'output_images/extracted_signs/'
            f'sign_{detected}.png'
        )

        cv2.imwrite(
            sign_path,
            extracted_sign
        )

        # -----------------------------------------------------
        # VISUALIZE EXTRACTED SIGN
        # -----------------------------------------------------

        plt.figure(figsize=(4,4))

        plt.imshow(
            cv2.cvtColor(
                extracted_sign,
                cv2.COLOR_BGR2RGB
            )
        )

        plt.title(
            f'Extracted Traffic Sign #{detected}'
        )

        plt.axis('off')

        plt.show()

    return output, detected


# ===================================================
# MAIN
# ===================================================

def main():

    os.makedirs(

        'output_images',

        exist_ok=True

    )

    input_folder = 'input_images'

    image_files = os.listdir(input_folder)

    for file_name in image_files:

        image_path = os.path.join(

            input_folder,

            file_name

        )

        img = cv2.imread(image_path)

        if img is None:

            print(f"Cannot load image: {file_name}")

            continue

        roi = extract_roi(img)

        blue_mask, red_mask = hsv_filter(roi)

        blue_clean = clean_mask(blue_mask)

        red_clean = clean_mask(red_mask)

        final_mask = cv2.add(

            blue_clean,

            red_clean

        )

        extract_output, extracted = extract_contours(

            final_mask,

            roi

        )

        contour_output, detected = semantic_contour_detection(

            final_mask,

            roi

        ) # contour_output is the final output image with detected contours drawn on it, detected is the number of contours that were detected as valid traffic signs after applying the semantic checks. By returning both the output image and the count of detected signs, we can visualize the results and also have a quantitative measure of how many signs were detected in each image.

        cv2.imwrite(

            f'output_images/extracted_contours_{file_name}',

            extract_output

        )

        cv2.imwrite(

            f'output_images/semantic_contours_{file_name}',

            contour_output

        )

        # ===================================================
        # PLOT 1: Inputs to semantic_contour_detection
        # ===================================================

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        plt.title('INPUT: ROI Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(final_mask, cmap='gray')
        plt.title('INPUT: Cleaned & Filtered Mask')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # ===================================================
        # PLOT 2: Full pipeline comparison
        # ===================================================

        plt.figure(figsize=(18, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        plt.title('ROI')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(final_mask, cmap='gray')
        plt.title('Filtered Mask')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(extract_output, cv2.COLOR_BGR2RGB)) # extract_output is the image with contours drawn from the extract_contours function, which represents the contours that were extracted based on the initial filtering criteria before applying the semantic checks. By plotting this image, we can visualize the contours that were identified as potential signs based on their geometric and color characteristics before we apply the more strict semantic checks in the next step. This allows us to see how many contours were initially extracted and how they compare to the final detected contours after applying the semantic checks.
        plt.title(f'Extracted Contours ({extracted})')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(contour_output, cv2.COLOR_BGR2RGB)) # contour_output is the final output image with detected contours drawn on it from the semantic_contour_detection function, which represents the contours that were accepted as valid traffic signs after applying the semantic checks. By plotting this image, we can visualize the final detected signs and compare them to the initially extracted contours to see how the semantic checks have refined the results. This allows us to see the effectiveness of the semantic contour detection step in filtering out false positives and keeping valid signs.
        plt.title(f'Semantic Contours ({detected})')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        print(f"Processed: {file_name}")

   
if __name__ == '__main__':

    main()

