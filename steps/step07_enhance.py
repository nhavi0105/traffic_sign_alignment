import cv2
import numpy as np


def is_valid_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # độ sắc nét
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # kích thước
    h, w = gray.shape

    if sharpness < 30:
        return False

    if h < 40 or w < 40:
        return False

    return True

def enhance_advanced(image):
    """
    Enhance traffic sign image:
    - Reduce noise (bilateral filter)
    - Sharpen edges
    - Improve contrast (CLAHE)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()

    # nếu ảnh tối → tăng contrast mạnh hơn
    if brightness < 100:
        clip = 1.8
    else:
        clip = 1.2

    # 1. Denoise 
    denoised = cv2.bilateralFilter(image, 5, 40, 40)

    # 2. Sharpen 
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 1.0)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)

    # 3. CLAHE 
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return enhanced
