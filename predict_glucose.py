import cv2
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import mean_absolute_error, r2_score

class GlucosePredictor:
    def __init__(self, model_path="glucose_model4.pkl", scaler_path="glucose_scaler4.pkl"):
        """Initialize predictor with proper feature name handling"""
        try:
            # Load model and scaler
            model_data = joblib.load(Path(model_path))
            self.model = model_data['model']  # Extract model from dictionary
            self.scaler = joblib.load(Path(scaler_path))
            
            # Get feature names
            self.feature_order = model_data.get('feature_names', [
                'contrast_d1_a0', 'contrast_d1_a45', 'contrast_d1_a90', 'contrast_d1_a135',
                'dissimilarity_d1_a0', 'dissimilarity_d1_a45', 'dissimilarity_d1_a90', 'dissimilarity_d1_a135',
                'homogeneity_d1_a0', 'homogeneity_d1_a45', 'homogeneity_d1_a90', 'homogeneity_d1_a135',
                'energy_d1_a0', 'energy_d1_a45', 'energy_d1_a90', 'energy_d1_a135',
                # ... (include all your features in the correct order)
                'speckle_contrast', 'intensity_mean', 'intensity_std', 'intensity_median'
            ])
            
            print("Model and scaler loaded successfully")
            print(f"Using feature order: {self.feature_order}")
            
        except Exception as e:
            raise ValueError(f"Error loading model/scaler: {str(e)}")

    def preprocess_image(self, image_path):
        """Load and preprocess image with validation"""
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to read image (may be corrupted)")
            
        img = cv2.resize(img, (256, 256))
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    def extract_features(self, image):
        """Extract features in the exact order expected by the scaler"""
        # Calculate GLCM features
        glcm = graycomatrix(image, distances=[1, 3, 5], 
                           angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                           levels=256)
        
        # Initialize features dictionary
        features = {}
        
        # Add GLCM features
        for i, angle in enumerate(['0', '45', '90', '135']):
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy']:
                for j, dist in enumerate([1, 3, 5]):
                    key = f"{prop}_d{dist}_a{angle}"
                    features[key] = graycoprops(glcm, prop)[j, i]
        
        # Add other features
        features.update({
            'speckle_contrast': np.std(image) / (np.mean(image) + 1e-10),
            'intensity_mean': np.mean(image),
            'intensity_std': np.std(image),
            'intensity_median': np.median(image)
        })
        
        # Return as DataFrame with correct column order
        return pd.DataFrame([features], columns=self.feature_order)

    def predict(self, image_path):
        """Make prediction with proper feature name handling"""
        try:
            # Preprocess and extract features
            img = self.preprocess_image(image_path)
            features_df = self.extract_features(img)
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Predict
            prediction = float(self.model.predict(features_scaled)[0])
            
            return {
                'status': 'success',
                'prediction': prediction,
                'image_path': str(Path(image_path).name)
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'image_path': str(Path(image_path).name)
            }

if __name__ == "__main__":
    try:
        predictor = GlucosePredictor()
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        exit(1)
    
    # Test images
    test_images = [
        (r"F:\8.Glucometer Setup IITH\test_data\400 mg-dL\aug_0_5593.png", 400),
        (r"F:\8.Glucometer Setup IITH\test_data\150 mg-dL\aug_0_1874.png", 150),
        (r"F:\8.Glucometer Setup IITH\test_data\200 mg-dL\aug_0_6346.png", 200)
    ]
    
    print("\nMaking predictions...\n")
    
    actual_values = []
    predicted_values = []
    
    for img_path, true_value in test_images:
        result = predictor.predict(img_path)
        
        if result['status'] == 'success':
            predicted = int(round(result['prediction']))
            error = abs(predicted - true_value)
            
            print(f"Image: {result['image_path']}")
            print(f"Predicted: {predicted} mg/dL")
            print(f"Actual: {true_value} mg/dL")
            print(f"Error: {error} mg/dL")
            
            actual_values.append(true_value)
            predicted_values.append(predicted)
        else:
            print(f"Error processing {result['image_path']}:")
            print(f"-> {result['message']}")
        
        print("-" * 50)
    
    # Calculate accuracy metrics
    # In your evaluation metrics section, replace with:

    if actual_values and predicted_values:
        print("\nEvaluation Metrics:")
        print("-" * 30)
        mae = mean_absolute_error(actual_values, predicted_values)
        r2 = r2_score(actual_values, predicted_values)
        
        # Calculate accuracy percentages for different thresholds
        thresholds = {
            '±10 mg/dL': 10,
            '±15 mg/dL': 15,
            '±20 mg/dL': 20
        }
        
        for name, threshold in thresholds.items():
            correct = sum(1 for a, p in zip(actual_values, predicted_values) if abs(a - p) <= threshold)
            accuracy = (correct / len(actual_values)) * 100
            print(f"Accuracy ({name}): {accuracy:.2f}%")
        
        print(f"\nMean Absolute Error (MAE): {mae:.2f} mg/dL")
        print(f"R-squared (R²) Score: {r2:.4f}")
        print("-" * 30)
    print('Prediction completed.')