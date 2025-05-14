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
import shiny_data

def train_and_evaluate_xgboostHOG(folder='original'):
    Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc = shiny_data.load_split_images()

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
    X_test_hog = extract_hog_features(Xmat_test)

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

    y_pred = xgb_model.predict(X_test_hog)
    accuracy = accuracy_score(y_test_enc, y_pred)
    f1 = f1_score(y_test_enc, y_pred, average='weighted')
    precision_per_class = precision_score(y_test_enc, y_pred, average=None)
    conf_matrix = confusion_matrix(y_test_enc, y_pred) 

    # Write metrics to CSV
    metrics_df = pd.DataFrame({
        'Folder': [folder],
        'Feature_Type': [PCA],
        'Accuracy': [accuracy],
        'F1_Score': [f1],
        'Precision_Class_0': [precision_per_class[0]],
        'Precision_Class_1': [precision_per_class[1]],
        'Precision_Class_2': [precision_per_class[2]],
        'Precision_Class_3': [precision_per_class[3]]
    })

    csv_file = 'xgboost_metrics.csv'
    if not os.path.exists(csv_file):
        metrics_df.to_csv(csv_file, index=False)
    else:
        metrics_df.to_csv(csv_file, mode='a', header=False, index=False)

    return xgb_model, accuracy, f1

    # Evaluate the model
    y_pred = xgb_model.predict(X_test_hog)
    accuracy = accuracy_score(y_test_enc, y_pred)
    fprec = precision_score(y_test_enc, y_pred, average='weighted')
    rec = recall_score(y_test_enc, y_pred, average='weighted')
    cm = confusion_matrix(y_test_enc, y_pred)

    # Print results
    print(f"Folder: {folder}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"F1 Score: {f1:.2%}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.title(f'Confusion Matrix - {folder}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{folder}_confusion_matrix.png')
    plt.close()

    return xgb_model, accuracy, f1

def train_and_evaluate_xgboostPCA(folder='original', n_components=100, save_pca=False, load_pca=True):
    Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc = shiny_data.load_split_images()

    X_train_flat = Xmat_train.reshape(Xmat_train.shape[0], -1)
    X_test_flat = Xmat_test.reshape(Xmat_test.shape[0], -1)

    pca_path = f'{folder}_pca.joblib'
    if load_pca and os.path.exists(pca_path):
        pca = joblib.load(pca_path)
    else:
        pca = PCA(n_components=n_components)
        pca.fit(X_train_flat)
        if save_pca:
            joblib.dump(pca, pca_path)

    X_train_pca = pca.transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)

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

    xgb_model.save_model('xgboost.json')

    y_pred = xgb_model.predict(X_test_pca)
    accuracy = accuracy_score(y_test_enc, y_pred)
    f1 = f1_score(y_test_enc, y_pred, average='weighted')
    precision_per_class = precision_score(y_test_enc, y_pred, average=None)
    conf_matrix = confusion_matrix(y_test_enc, y_pred) 

    # Write metrics to CSV
    metrics_df = pd.DataFrame({
        'Folder': [folder],
        'Feature_Type': 'PCA',
        'Accuracy': [accuracy],
        'F1_Score': [f1],
        'Precision_Class_0': [precision_per_class[0]],
        'Precision_Class_1': [precision_per_class[1]],
        'Precision_Class_2': [precision_per_class[2]],
        'Precision_Class_3': [precision_per_class[3]]
    })

    csv_file = 'xgboost_metrics.csv'
    if not os.path.exists(csv_file):
        metrics_df.to_csv(csv_file, index=False)
    else:
        metrics_df.to_csv(csv_file, mode='a', header=False, index=False)

    return xgb_model, accuracy, f1

train_and_evaluate_xgboostPCA(folder='Base', save_pca=True, load_pca=False)

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

# # Call the function
# test_hyperparameters()