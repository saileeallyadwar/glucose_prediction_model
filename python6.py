# First install required packages if needed (run these commands in your terminal/command prompt)
"""
python -m pip install lightgbm xgboost scikit-learn opencv-python scikit-image joblib tqdm pandas numpy
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')

class GlucosePredictor:
    def __init__(self, model_type='xgb', tuning_mode='random'):
        """
        Initialize predictor with model type and tuning approach
        model_type: 'rf' (Random Forest), 'xgb' (XGBoost), 'lgbm' (LightGBM), 'svr' (SVR)
        tuning_mode: 'grid', 'random', or None
        """
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_type = model_type.lower()
        self.tuning_mode = tuning_mode.lower() if tuning_mode else None
        self.best_params_ = None
        
    def is_image_valid(self, image_path):
        """Check if image is readable and valid"""
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False
            if img.size == 0:
                return False
            if np.all(img == img[0,0]):  # Check if image is all one color
                return False
            return True
        except Exception as e:
            print(f"Validation failed for {image_path}: {str(e)}")
            return False
    
    def preprocess_image(self, image_path):
        """Load and preprocess image with enhanced checks"""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        if img.size == 0:
            raise ValueError(f"Empty image: {image_path}")
        
        img = cv2.resize(img, (256, 256))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img
    
    def extract_features(self, image_path):
        """Enhanced feature extraction with LBP and additional features"""
        try:
            img = self.preprocess_image(image_path)
        except Exception as e:
            raise ValueError(f"Image preprocessing failed: {str(e)}")
        
        features = {}
        
        # 1. GLCM Features (multiple distances)
        for distance in [1, 3, 5]:
            glcm = graycomatrix(img, 
                              distances=[distance], 
                              angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                              levels=256,
                              symmetric=True,
                              normed=True)
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
                features.update({
                    f'{prop}_d{distance}_a{angle}': graycoprops(glcm, prop)[0, i]
                    for i, angle in enumerate(['0', '45', '90', '135'])
                })
        
        # 2. LBP Features (multiple radii)
        for radius in [1, 3, 5]:
            n_points = 8 * radius
            lbp = local_binary_pattern(img, n_points, radius, method='uniform')
            hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), 
                                 range=(0, n_points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-10)
            features.update({f'lbp_r{radius}_{i}': val for i, val in enumerate(hist)})
        
        # 3. Speckle Statistics
        features.update({
            'speckle_contrast': np.std(img) / (np.mean(img) + 1e-10),
            'intensity_mean': np.mean(img),
            'intensity_std': np.std(img),
            'intensity_median': np.median(img),
            'intensity_skewness': pd.Series(img.flatten()).skew(),
            'intensity_kurtosis': pd.Series(img.flatten()).kurtosis(),
        })
        
        # 4. Histogram features
        hist = cv2.calcHist([img], [0], None, [16], [0, 256])
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-10)
        features.update({f'hist_{i}': val[0] for i, val in enumerate(hist)})
        
        return features
    
    def _get_model(self):
        """Return base model based on model_type"""
        if self.model_type == 'rf':
            return RandomForestRegressor(random_state=42, n_jobs=-1)
        elif self.model_type == 'xgb':
            return XGBRegressor(random_state=42, n_jobs=-1)
        elif self.model_type == 'lgbm':
            return LGBMRegressor(random_state=42, n_jobs=-1)
        elif self.model_type == 'svr':
            return SVR()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _get_param_grid(self):
        """Return parameter grid based on model_type and tuning_mode"""
        if self.model_type == 'rf':
            if self.tuning_mode == 'grid':
                return {
                    'n_estimators': [100, 150, 200],
                    'max_depth': [10, 12, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            else:  # random
                return {
                    'n_estimators': randint(50, 300),
                    'max_depth': randint(5, 20),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10),
                    'max_features': ['sqrt', 'log2']
                }
        
        elif self.model_type == 'xgb':
            if self.tuning_mode == 'grid':
                return {
                    'n_estimators': [100, 150, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            else:
                return {
                    'n_estimators': randint(50, 300),
                    'max_depth': randint(3, 10),
                    'learning_rate': uniform(0.01, 0.3),
                    'subsample': uniform(0.6, 0.4),
                    'colsample_bytree': uniform(0.6, 0.4),
                    'gamma': uniform(0, 0.5)
                }
        
        elif self.model_type == 'lgbm':
            if self.tuning_mode == 'grid':
                return {
                    'num_leaves': [15, 31, 63],
                    'max_depth': [-1, 5, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [100, 200, 300],
                    'min_child_samples': [10, 20, 30]
                }
            else:
                return {
                    'num_leaves': randint(10, 100),
                    'max_depth': randint(3, 15),
                    'learning_rate': uniform(0.01, 0.3),
                    'n_estimators': randint(50, 500),
                    'min_child_samples': randint(5, 50)
                }
        
        elif self.model_type == 'svr':
            return {
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.5, 1.0],
                'gamma': ['scale', 'auto']
            }

    def train(self, data_dir, test_size=0.2, random_state=42, cv=3):
        """Enhanced training with hyperparameter tuning"""
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        features, labels = [], []
        
        # Process each concentration folder
        conc_folders = [f for f in data_path.iterdir() if f.is_dir()]
        if not conc_folders:
            raise ValueError("No concentration folders found")
            
        for conc_folder in tqdm(conc_folders, desc="Processing folders"):
            try:
                glucose_level = float(conc_folder.name.split()[0])
            except ValueError:
                print(f"Skipping folder with invalid name: {conc_folder.name}")
                continue
                
            # Process valid images
            valid_images = [img for img in conc_folder.glob('*.*') 
                          if img.suffix.lower() in ('.png', '.jpg', '.jpeg') 
                          and self.is_image_valid(img)]
            
            if not valid_images:
                print(f"No valid images found in {conc_folder.name}")
                continue
                
            for img_path in tqdm(valid_images, desc=f"{conc_folder.name}", leave=False):
                try:
                    features.append(self.extract_features(img_path))
                    labels.append(glucose_level)
                except Exception as e:
                    print(f"Skipped {img_path.name}: {str(e)}")
                    continue
        
        if not features:
            raise ValueError("No valid images found for training")
        
        # Create and scale features
        features_df = pd.DataFrame(features)
        self.feature_names = features_df.columns.tolist()
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(features_df)
        y = np.array(labels)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Initialize model and parameter grid
        model = self._get_model()
        param_grid = self._get_param_grid()
        
        # Perform hyperparameter tuning
        if self.tuning_mode == 'grid':
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=1
            )
        elif self.tuning_mode == 'random':
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=20,
                cv=cv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=1,
                random_state=random_state
            )
        else:
            # No tuning - use default parameters
            self.model = model
            self.model.fit(X_train, y_train)
            return self
        
        print(f"\nStarting {self.tuning_mode} search for {self.model_type}...")
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        self.best_params_ = search.best_params_
        
        print("\nBest parameters found:")
        for param, value in self.best_params_.items():
            print(f"{param}: {value}")
        
        # Evaluate
        for name, X, y in [("Training", X_train, y_train), ("Validation", X_test, y_test)]:
            y_pred = self.model.predict(X)
            print(f"\n{name} Metrics:")
            print(f"MAE: {mean_absolute_error(y, y_pred):.2f} mg/dL")
            print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.2f} mg/dL")
            print(f"R²: {r2_score(y, y_pred):.4f}")
            print(f"Samples: {len(y)}")
        
        return self

    def predict(self, image_path):
        """Make prediction on a single image"""
        try:
            if not self.model or not self.scaler:
                raise ValueError("Model not trained or loaded")
            
            if not Path(image_path).exists():
                return {'status': 'error', 'message': 'Image file not found'}
            
            features = self.extract_features(image_path)
            features_df = pd.DataFrame([features])
            X = self.scaler.transform(features_df)
            prediction = self.model.predict(X)[0]
            
            return {
                'status': 'success',
                'image_path': str(image_path),
                'prediction': prediction
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def save(self, model_path="model.pkl", scaler_path="scaler.pkl"):
        try:
            # Validate we have trained components
            if self.model is None:
                raise ValueError("Model has not been trained yet")
            if self.scaler is None:
                raise ValueError("Scaler has not been initialized")
            if not hasattr(self, 'feature_names'):
                raise ValueError("Feature names not available")
            
            # Save model components
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'best_params': self.best_params_
            }
            joblib.dump(model_data, model_path)
            
            # Save scaler separately
            joblib.dump(self.scaler, scaler_path)
            
            print(f"Successfully saved model to {model_path}")
            print(f"Successfully saved scaler to {scaler_path}")
            
        except Exception as e:
            print(f"Failed to save model/scaler: {str(e)}")
            raise

    def load(self, model_path="model.pkl", scaler_path="scaler.pkl"):
        """Load model and scaler from separate files"""
        try:
            # Load model components
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data.get('model_type', 'xgb')
            self.best_params_ = model_data.get('best_params', None)
            
            # Load scaler
            self.scaler = joblib.load(scaler_path)
            
            print(f"Successfully loaded model from {model_path}")
            print(f"Successfully loaded scaler from {scaler_path}")
            return self
            
        except FileNotFoundError:
            raise FileNotFoundError("Model or scaler file not found")
        except Exception as e:
            raise ValueError(f"Error loading model/scaler: {str(e)}")


if __name__ == "__main__":
    # Initialize predictor with XGBoost and random search tuning
    predictor = GlucosePredictor(model_type='xgb', tuning_mode='random')
    
    # Define paths (update these with your actual paths)
    dataset_path = Path(r"F:\8.Glucometer Setup IITH\train_data")
    testset_path = Path(r"F:\8.Glucometer Setup IITH\test_data")
    model_path = r"F:\8.Glucometer Setup IITH\glucose_model4.pkl"
    scaler_path = r"F:\8.Glucometer Setup IITH\glucose_scaler4.pkl"
    
    try:
        # 1. Verify datasets exist
        if not dataset_path.exists():
            raise FileNotFoundError(f"Training dataset path does not exist: {dataset_path}")
        if not testset_path.exists():
            print("Warning: Test dataset path not found - will only evaluate on training data")
        
        print(f"Starting training process with {predictor.model_type} model...")
        
        # 2. Train the model
        predictor.train(dataset_path, cv=3)
        
        # 3. Save the trained model and scaler
        print("\nSaving trained model...")
        predictor.save(model_path=model_path, scaler_path=scaler_path)
        
        # 4. Load the saved model for verification
        print("\nVerifying model loading...")
        loaded_predictor = GlucosePredictor()
        loaded_predictor.load(model_path=model_path, scaler_path=scaler_path)
        print(f"Model loaded successfully! Model type: {loaded_predictor.model_type}")
        
        # 5. Evaluate on test set if available
        if testset_path.exists():
            print("\nEvaluating on test set...")
            test_features = []
            test_labels = []
            
            # Process test images
            for conc_folder in testset_path.iterdir():
                if not conc_folder.is_dir():
                    continue
                
                try:
                    glucose_level = float(conc_folder.name.split()[0])
                except ValueError:
                    continue
                
                for img_path in conc_folder.glob('*.*'):
                    if img_path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                        continue
                    
                    try:
                        features = loaded_predictor.extract_features(img_path)
                        test_features.append(features)
                        test_labels.append(glucose_level)
                    except Exception as e:
                        print(f"Skipped {img_path.name}: {str(e)}")
                        continue
            
            if test_features:
                # Prepare test data
                test_df = pd.DataFrame(test_features)
                X_test = loaded_predictor.scaler.transform(test_df)
                y_test = np.array(test_labels)
                
                # Make predictions
                y_pred = loaded_predictor.model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                print("\nTest Set Evaluation Metrics:")
                print(f"- MAE: {mae:.2f} mg/dL")
                print(f"- MSE: {mse:.2f} mg²/dL²")
                print(f"- RMSE: {rmse:.2f} mg/dL")
                print(f"- R² Score: {r2:.4f}")
                
                # Sample some predictions
                print("\nSample Predictions:")
                sample_indices = np.random.choice(len(y_test), min(5, len(y_test)), replace=False)
                for idx in sample_indices:
                    print(f"Predicted: {int(round(y_pred[idx]))} mg/dL | Actual: {int(y_test[idx])} mg/dL")
            else:
                print("No valid test images found for evaluation")
        
        # 6. Example single image prediction
        test_image = Path(r"F:\8.Glucometer Setup IITH\test_image.png")
        if test_image.exists():
            result = loaded_predictor.predict(test_image)
            if result['status'] == 'success':
                print(f"\nSingle Image Prediction:")
                print(f"- Image: {result['image_path']}")
                print(f"- Predicted Glucose: {int(round(result['prediction']))} mg/dL")
            else:
                print(f"\nPrediction failed: {result['message']}")
    
    except FileNotFoundError as e:
        print(f"\nFile error: {str(e)}")
    except ValueError as e:
        print(f"\nTraining error: {str(e)}")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
    finally:
        print("\nProcess completed")