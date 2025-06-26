import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def compute_glcm_image_feature(img, feature='contrast', distance=1, angle=0):
    win_size = 9  # Must be odd
    pad = win_size // 2
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(pad, pad + img.shape[0]):
        for j in range(pad, pad + img.shape[1]):
            window = img_padded[i-pad:i+pad+1, j-pad:j+pad+1]
            glcm = graycomatrix(window, 
                                distances=[distance], 
                                angles=[angle], 
                                levels=256, 
                                symmetric=True, 
                                normed=True)
            value = graycoprops(glcm, feature)[0, 0]
            output[i - pad, j - pad] = value
    
    # Normalize and convert to image
    output_img = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
    output_img = output_img.astype(np.uint8)
    return output_img

# Example usage
gray_img = cv2.imread(r"F:\8.Glucometer Setup IITH\download.jpeg", cv2.IMREAD_GRAYSCALE)
glcm_contrast_img = compute_glcm_image_feature(gray_img, feature='contrast', angle=0)
cv2.imwrite("glcm_contrast_output.png", glcm_contrast_img)
