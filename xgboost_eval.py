
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
import joblib
import os
import shiny_data

def test_augmented_xgboost():
    _, _, Xmat_test, _, _, y_test_enc = shiny_data.load_split_images()
    X_test_flat = Xmat_test.reshape(Xmat_test.shape[0], -1)

    # Load PCA
    pca = joblib.load('Base_pca.joblib')
    
    # Load pre-trained XGBoost model
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('xgboost.json')

    blur_sizes = [0]
    contrast_factors = [1, 1.25, 1.5, 1.75, 2]

    metrics_list = []

    for blur_size in blur_sizes:
        for contrast_factor in contrast_factors:
            # Apply augmentations
            X_test_aug = shiny_data.adjust_contrast(shiny_data.apply_blur(Xmat_test, blur_size), contrast_factor)
            X_test_aug_flat = X_test_aug.reshape(X_test_aug.shape[0], -1)
            
            # Apply PCA
            X_test_aug_pca = pca.transform(X_test_aug_flat)

            # Predict
            y_pred = xgb_model.predict(X_test_aug_pca)

            # Calculate metrics
            accuracy = accuracy_score(y_test_enc, y_pred)
            f1 = f1_score(y_test_enc, y_pred, average='weighted')
            precision_per_class = precision_score(y_test_enc, y_pred, average=None)

            # Append metrics to list
            metrics_list.append({
                'Blur_Size': blur_size,
                'Contrast_Factor': contrast_factor,
                'Accuracy': accuracy,
                'F1_Score': f1,
                'Precision_Class_z0': precision_per_class[0],
                'Precision_Class_z1': precision_per_class[1],
                'Precision_Class_z2': precision_per_class[2],
                'Precision_Class_z3': precision_per_class[3]
            })

            print(f"Blur {blur_size}, Contrast {contrast_factor}")
            print(f"Accuracy {accuracy}, ")

    # Create DataFrame and save to CSV
    metrics_df = pd.DataFrame(metrics_list)
    csv_file = 'xgboost_augmented_test_metrics.csv'
    metrics_df.to_csv(csv_file, mode='a', header=None, index=False)
    print(f"Metrics appended to {csv_file}")

if __name__ == "__main__":
    test_augmented_xgboost()