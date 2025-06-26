import os
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import shutil
import logging
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
original_data_dir = r"F:\8. Glucometer Setup IITH\original_dataset"
augmented_data_dir = r'F:\8. Glucometer Setup IITH\augemented_dataset'
train_dir = r'F:\8. Glucometer Setup IITH\train_data'
test_dir = r'F:\8. Glucometer Setup IITH\test_data'

# Create directories if they don't exist
os.makedirs(augmented_data_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

def is_valid_image(file_path):
    """Check if a file is a valid image"""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError, UnidentifiedImageError) as e:
        logger.warning(f"Invalid image file {file_path}: {str(e)}")
        return False

def load_image_robust(file_path):
    """Try multiple methods to load an image"""
    try:
        # Method 1: Try PIL first
        try:
            img = Image.open(file_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)
        except (UnidentifiedImageError, SyntaxError) as e:
            logger.warning(f"PIL failed for {file_path}, trying OpenCV: {str(e)}")
            
        # Method 2: Try OpenCV if PIL fails
        try:
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("OpenCV returned None")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            logger.warning(f"OpenCV failed for {file_path}: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error loading {file_path}: {str(e)}")
        return None

# Define augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Process each concentration folder
for concentration in os.listdir(original_data_dir):
    conc_dir = os.path.join(original_data_dir, concentration)
    if not os.path.isdir(conc_dir):
        continue
    
    # Create corresponding augmented folder
    aug_conc_dir = os.path.join(augmented_data_dir, concentration)
    os.makedirs(aug_conc_dir, exist_ok=True)
    
    # Process each image in the concentration folder
    for img_name in os.listdir(conc_dir):
        img_path = os.path.join(conc_dir, img_name)
        
        # Skip if not a valid image
        if not is_valid_image(img_path):
            continue
            
        try:
            # Load image using robust method
            img_array = load_image_robust(img_path)
            if img_array is None:
                continue
                
            img_array = np.expand_dims(img_array, axis=0)
            
            # Generate augmented images
            i = 0
            for batch in datagen.flow(img_array, batch_size=1,
                                   save_to_dir=aug_conc_dir,
                                   save_prefix='aug',
                                   save_format='png'):
                i += 1
                if i >= 5:  # Generate 5 augmented images per original
                    break
                    
        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")
            continue

# Now split augmented data into train/test (80/20 split)
for concentration in os.listdir(augmented_data_dir):
    conc_dir = os.path.join(augmented_data_dir, concentration)
    
    # Create train/test folders for this concentration
    train_conc_dir = os.path.join(train_dir, concentration)
    test_conc_dir = os.path.join(test_dir, concentration)
    os.makedirs(train_conc_dir, exist_ok=True)
    os.makedirs(test_conc_dir, exist_ok=True)
    
    # Get all images for this concentration
    all_images = [img for img in os.listdir(conc_dir) if img.lower().endswith('.png')]
    
    # Split 80/20
    train_images, test_images = train_test_split(all_images, test_size=0.2, random_state=42)
    
    # Copy to respective folders
    for img in train_images:
        src = os.path.join(conc_dir, img)
        dst = os.path.join(train_conc_dir, img)
        shutil.copy(src, dst)
    
    for img in test_images:
        src = os.path.join(conc_dir, img)
        dst = os.path.join(test_conc_dir, img)
        shutil.copy(src, dst)
