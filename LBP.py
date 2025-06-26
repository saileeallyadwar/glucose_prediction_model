import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from pathlib import Path

def extract_and_save_lbp(image_path, radius=3, n_points=24, save_result=True):
    # Load grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Resize (optional)
    image = cv2.resize(image, (256, 256))

    # Compute LBP
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')

    # Normalize for image saving (convert float to 0-255 grayscale)
    lbp_image = (lbp / lbp.max()) * 255
    lbp_image = lbp_image.astype(np.uint8)

    # Save result image
    if save_result:
        output_path = Path(image_path).with_name(f"{Path(image_path).stem}_LBP.png")
        cv2.imwrite(str(output_path), lbp_image)
        print(f"LBP image saved to: {output_path}")

    # Histogram (26 bins for uniform patterns)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-10)  # Normalize to sum to 1

    return hist

# Example usage
lbp_hist = extract_and_save_lbp("F:\8.Glucometer Setup IITH\download.jpeg")
print("LBP feature vector:", lbp_hist)
