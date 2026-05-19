import os
import cv2
from steps.step07_enhance import enhance_advanced, is_valid_image

input_folder = "/Users/nguyenhavi/Documents/traffic_sign_alignment-nhuy-work/output_images/extracted_signs"
output_folder = "/Users/nguyenhavi/Documents/traffic_sign_alignment-nhuy-work/output_images/enhanced_signs"

os.makedirs(output_folder, exist_ok=True)

print("Starting enhancement...\n")

for filename in os.listdir(input_folder):

    if not (filename.endswith(".png") or filename.endswith(".jpg")):
        continue

    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    compare_path = os.path.join(output_folder, "compare_" + filename)

    img = cv2.imread(input_path)

    if img is None:
        print(f"Cannot read: {filename}")
        continue

    # ❗ Filter ảnh xấu
    if not is_valid_image(img):
        print(f"Skip (bad quality): {filename}")
        continue

    # Enhance
    enhanced = enhance_advanced(img)

    # Resize ảnh gốc cho bằng enhanced
    img_resized = cv2.resize(img, (enhanced.shape[1], enhanced.shape[0]))

    # Concat đúng
    compare = cv2.hconcat([img_resized, enhanced])

    # Save
    cv2.imwrite(output_path, enhanced)
    cv2.imwrite(compare_path, compare)

    print(f"Processed: {filename}")

print("\nDone all images!")