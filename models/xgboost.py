import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import xgboost as xg
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import time
import joblib
import itertools
import csv
from skimage.feature import hog
import data_preprocessing

#========HOG TRAINING==========
def train_and_evaluate_xgboostHOG(folder='original'):
    Xmat_train, Xmat_val, _, _, _, y_train_enc, y_val_enc, _, _, _ = data_preprocessing.load_split_images()

    def extract_hog_features(images):
        hog_features = []
        for img in images:
            # Convert to grayscale if it's RGB
            if img.shape[-1] == 3:
                img = np.mean(img, axis=-1)
            features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
            hog_features.append(features)
        return np.array(hog_features)

    X_train_hog = extract_hog_features(Xmat_train)
    X_val_hog = extract_hog_features(Xmat_val)

    # Initialize and train XGBoost model
    xgb_model = xg.XGBClassifier(
        objective='multi:softmax',
        num_class=4,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=1,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=6,
        device='cuda'
    )

    print("Start Training")

    xgb_model.fit(X_train_hog, y_train_enc)
    print("Finish training")

    y_pred = xgb_model.predict(X_val_hog)
    accuracy = accuracy_score(y_val_enc, y_pred)
    f1 = f1_score(y_val_enc, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_val_enc, y_pred) 

    return xgb_model, accuracy, f1, conf_matrix

#==========PCA TRAINING=========

def train_and_evaluate_xgboostPCA(folder='original', n_components=100, save_pca=False, load_pca=True):
    Xmat_train, Xmat_val, _, _, _, y_train_enc, y_val_enc, _, _, _ = data_preprocessing.load_split_images()

    X_train_flat = Xmat_train.reshape(Xmat_train.shape[0], -1)
    X_val_flat = Xmat_val.reshape(Xmat_val.shape[0], -1)

    pca_path = f'{folder}_pca.joblib'
    if load_pca and os.path.exists(pca_path):
        pca = joblib.load(pca_path)
    else:
        pca = PCA(n_components=n_components)
        pca.fit(X_train_flat)
        if save_pca:
            joblib.dump(pca, pca_path)

    X_train_pca = pca.transform(X_train_flat)
    X_val_pca = pca.transform(X_val_flat)

    xgb_model = xg.XGBClassifier(
        objective='multi:softmax',
        num_class=4,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=1,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=6,
        device='cuda'
    )

    xgb_model.fit(X_train_pca, y_train_enc)
    print("Finish training")

    xgb_model.save_model('models/xgboost.json')

    y_pred = xgb_model.predict(X_val_pca)
    accuracy = accuracy_score(y_val_enc, y_pred)
    f1 = f1_score(y_val_enc, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_val_enc, y_pred) 

    return xgb_model, accuracy, f1, conf_matrix

#======== HYPERPARAMETER OPTIMISATION======

# def test_hyperparameters():
#     # Define parameter ranges
#     param_ranges = {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [3, 5, 7],
#         'learning_rate': [0.01, 0.1, 0.3],
#         'subsample': [0.8, 1.0]
#     }

#     # Generate all combinations of parameters
#     param_combinations = list(itertools.product(*param_ranges.values()))

#     # Prepare CSV file for results
#     with open('xgboost_results.csv', 'w', newline='') as csvfile:
#         fieldnames = list(param_ranges.keys()) + ['accuracy', 'f1_score', 'training_time']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         csvfile.flush()

#         # Test each combination
#         for params in param_combinations:
#             try:
#                 n_estimators, max_depth, learning_rate, subsample = params
                
#                 model, accuracy, f1, training_time = train_and_evaluate_xgboost(
#                     folder='original',
#                     n_estimators=n_estimators,
#                     max_depth=max_depth,
#                     learning_rate=learning_rate,
#                     subsample=subsample,
#                     load_pca=True
#                 )

#                 # Write results to CSV
#                 writer.writerow({
#                     'n_estimators': n_estimators,
#                     'max_depth': max_depth,
#                     'learning_rate': learning_rate,
#                     'subsample': subsample,
#                     'accuracy': accuracy,
#                     'f1_score': f1,
#                     'training_time': training_time
#                 })
#                 csvfile.flush()

#                 print(f"Completed: {params}")

#             except Exception as e:
#                 print(f"Error with parameters {params}: {str(e)}")
#                 continue

#=============== RUN TRAINING  ============

if __name__ == "__main__":
    # Example usage: Train with HOG features
    print("Running XGBoost with HOG features...")
    hog_model, hog_acc, hog_f1 = train_and_evaluate_xgboostHOG(folder='original')
    print(f"HOG Accuracy: {hog_acc:.4f} | F1: {hog_f1:.4f}")

    # Example usage: Train with PCA features
    print("Running XGBoost with PCA features...")
    pca_model, pca_acc, pca_f1, pca_conf = train_and_evaluate_xgboostPCA(folder='original')
    print(f"PCA Accuracy: {pca_acc:.4f} | F1: {pca_f1:.4f}")
    print(pca_conf)