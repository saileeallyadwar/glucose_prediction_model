import cv2
import numpy as np

def speckle_contrast_image(img, window_size=15):
    pad = window_size // 2
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    contrast_map = np.zeros_like(img, dtype=np.float32)

    for i in range(pad, pad + img.shape[0]):
        for j in range(pad, pad + img.shape[1]):
            window = img_padded[i-pad:i+pad+1, j-pad:j+pad+1]
            mean = np.mean(window)
            std = np.std(window)
            contrast = std / (mean + 1e-10)
            contrast_map[i - pad, j - pad] = contrast

    # Normalize to 0-255 for display
    contrast_img = cv2.normalize(contrast_map, None, 0, 255, cv2.NORM_MINMAX)
    contrast_img = contrast_img.astype(np.uint8)
    return contrast_img

# Example usage
gray = cv2.imread(r"F:\8.Glucometer Setup IITH\download.jpeg", cv2.IMREAD_GRAYSCALE)
contrast_img = speckle_contrast_image(gray)
cv2.imwrite("speckle_contrast_visual.png", contrast_img)
