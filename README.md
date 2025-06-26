# Glucose Level Prediction from Images  

A machine learning model that predicts blood glucose levels (mg/dL) from biological sample images.  

## **Installation**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/glucose-prediction.git
   cd glucose-prediction
   ```
Install dependencies:

```bash
pip install -r requirements.txt
```
Dataset Structure
Place images in folders named by their glucose concentration:

```bash
data/
├── train/
│   ├── 100mg/      # Folder for 100 mg/dL samples
│   │   ├── img1.png
│   │   └── ...
│   └── 200mg/
│       └── ...
└── test/
    └── ...         # Same structure as train/
```
Usage
1. Training the Model
bash
python train.py --data_dir ./data/train --model_type xgb --tuning_mode random
Arguments:

--model_type: xgb (default), rf, svr, or lgbm

--tuning_mode: grid, random, or none

2. Running Predictions
```bash
python predict.py --image sample.png --model glucose_model.pkl --scaler glucose_scaler.pkl
```
3. Evaluating on Test Set
```bash
python evaluate.py --data_dir ./data/test
```
Outputs
----
- Trained model: glucose_model.pkl
- Feature scaler: glucose_scaler.pkl
- Metrics: MAE, RMSE, R²

Feature Extraction Details
---
Feature	Description	Key Parameters
 - GLCM	Texture analysis (contrast, energy, etc.)	Distances: [1,3,5], Angles: [0°,45°,90°,135°]
 - LBP	Local texture patterns	Radius: 1,3,5; Points: 8×radius
 - Speckle Stats	Intensity distribution (mean, std, etc.)	
text

---

### **Key Files in Project**  
| File          | Purpose                                  |
|---------------|------------------------------------------|
| `train.py`    | Trains ML models with hyperparameter tuning. |  
| `predict.py`  | Predicts glucose from a single image.    |  
| `evaluate.py` | Tests model performance on a dataset.    |  
| `requirements.txt` | Lists Python dependencies.           |  

---

### **How to Contribute**  
1. **Report issues** for bugs/unexpected results.  
2. **Suggest features**:  
   - Add new image preprocessing methods.  
   - Integrate deep learning models.  
3. **Improve documentation**.  

**Contact**: saileeallyadwar51@gmail.com  

--- 

This `README.md` provides a **clear, structured guide** for users to run the project. Let me know if you'd like to add more sections!
