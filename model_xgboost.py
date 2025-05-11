import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
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

def train_and_evaluate_xgboost(folder='original', n_components=100, save_pca=False, load_pca=True, n_estimators=100, max_depth=5, learning_rate=0.1, subsample=1):
    base_path = f"{folder}"
    categories = ['immune', 'tumour', 'stromal', 'other']

    def load_and_preprocess_images(category):
        category_path = os.path.join(base_path, category)
        images = []
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            images.append(img_array.flatten())  # Flatten the image
        return np.array(images)

    X = []
    y = []
    for category in categories:
        category_images = load_and_preprocess_images(category)
        X.extend(category_images)
        y.extend([category] * len(category_images))
    
    print("Images loaded")

    X = np.array(X)
    y = np.array(y)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    if load_pca:
        pca = joblib.load(f'{folder}_pca.joblib')
    else:
        pca = PCA(n_components=n_components)
        pca.fit(X)
        if save_pca:
            joblib.dump(pca, f'{folder}_pca.joblib')

    X_pca = pca.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.475, random_state=42)

    print("Split complete")


    # Initialize and train XGBoost model
    xgb_model = xg.XGBClassifier(
        objective='multi:softmax',
        num_class=4,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=6,
        device='cuda'
    )

    print("Start training")

    start_time = time.time()
    xgb_model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    print("Finish training")

    # Evaluate the model
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

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

    return xgb_model, accuracy, f1, training_time

def test_hyperparameters():
    # Define parameter ranges
    param_ranges = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0]
    }

    # Generate all combinations of parameters
    param_combinations = list(itertools.product(*param_ranges.values()))

    # Prepare CSV file for results
    with open('xgboost_results.csv', 'w', newline='') as csvfile:
        fieldnames = list(param_ranges.keys()) + ['accuracy', 'f1_score', 'training_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.flush()

        # Test each combination
        for params in param_combinations:
            try:
                n_estimators, max_depth, learning_rate, subsample = params
                
                model, accuracy, f1, training_time = train_and_evaluate_xgboost(
                    folder='original',
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    load_pca=True
                )

                # Write results to CSV
                writer.writerow({
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'subsample': subsample,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'training_time': training_time
                })
                csvfile.flush()

                print(f"Completed: {params}")

            except Exception as e:
                print(f"Error with parameters {params}: {str(e)}")
                continue

# Call the function
test_hyperparameters()