#import libraries
from PIL import Image, ImageFilter, ImageEnhance
import gc
# import tensorflow as tf
# from tensorflow.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt
from pathlib import Path
import random
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xg
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import time

print("libraries imported")

#-------------------IMPORT & PREPROCESS-------------------

base_path = "data/100"

# Get tumour file paths and shuffle
tumour_files = []
tumour_dirs = [
    "Invasive_Tumor",
    "Prolif_Invasive_Tumor",
    "T_Cell_and_Tumor_Hybrid"
]

for dir_name in tumour_dirs:
    dir_path = os.path.join(base_path, dir_name)
    if os.path.isdir(dir_path):
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
        tumour_files.extend(files)

random.shuffle(tumour_files)

# Get immune file paths and shuffle
immune_files = []
immune_dirs = [
    "CD4+_T_Cells", "CD4+_T_Cells",
    "CD8+_T_Cells",
    "B_Cells",
    "Mast_Cells",
    "Macrophages_1",
    "Macrophages_2",
    "LAMP3+_DCs",
    "IRF7+_DCs"
]

for dir_name in immune_dirs:
    dir_path = os.path.join(base_path, dir_name)
    if os.path.isdir(dir_path):
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
        immune_files.extend(files)

random.shuffle(immune_files)


# Get stromal file paths and shuffle
stromal_files = []
stromal_dirs = [
    "Stromal",
    "Stromal_and_T_Cell_Hybrid",
    "Perivascular-Like"
]

for dir_name in stromal_dirs:
    dir_path = os.path.join(base_path, dir_name)
    if os.path.isdir(dir_path):
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
        stromal_files.extend(files)

random.shuffle(stromal_files)

# Get other file paths and shuffle
other_files = []
other_dirs = [
    "Endothelial",
    "Myoepi_ACTA2+",
    "Myoepi_KRT15+",
    "DCIS_1",
    "DCIS_2"
]

for dir_name in stromal_dirs:
    dir_path = os.path.join(base_path, dir_name)
    if os.path.isdir(dir_path):
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
        other_files.extend(files)

random.shuffle(other_files)

def load_image(img_path):
    img = Image.open(img_path).convert('RGB')
    return np.array(img)

tumour_imgs = [load_image(f) for f in tumour_files]
print("tumour loaded")

immune_imgs = [load_image(f) for f in immune_files]
print("immune loaded")

stromal_imgs = [load_image(f) for f in stromal_files]
print("stromal loaded")

other_imgs = [load_image(f) for f in other_files]
print("other loaded")

#-------------------TEST TRAIN SPLIT--------------------

def resize_images(images, size=(224,224)):
    resized = []
    for img in images:
        pil_img = Image.fromarray(img)
        resized.append(np.array(pil_img.resize(size)))
    return np.array(resized)

tumour_train_ind = 80
tumour_test_ind = 20
tumour_total = tumour_train_ind + tumour_test_ind

immune_train_ind = 80
immune_test_ind = 20
immune_total = immune_train_ind + immune_test_ind

stromal_train_ind = 80
stromal_test_ind = 20
stromal_total = stromal_train_ind + stromal_test_ind

other_train_ind = 80
other_test_ind = 20
other_total = other_train_ind + other_test_ind

imgs_train = immune_imgs[:immune_train_ind] + tumour_imgs[:tumour_train_ind] + stromal_imgs[:stromal_train_ind] + other_imgs[:other_train_ind]
imgs_test = immune_imgs[immune_train_ind:immune_total] + tumour_imgs[tumour_train_ind:tumour_total] + stromal_imgs[stromal_train_ind:stromal_total] + other_imgs[other_train_ind:other_total]

imgs_train = resize_images(imgs_train)
imgs_test = resize_images(imgs_test)

Xmat_train = np.stack(imgs_train, axis=0)
Xmat_test = np.stack(imgs_test, axis=0)

y_train = ['Immune'] * immune_train_ind + ['Tumour'] * tumour_train_ind + ['Stromal'] * stromal_train_ind + ['Other'] * other_train_ind
y_test = ['Immune'] * immune_test_ind + ['Tumour'] * tumour_test_ind + ['Stromal'] * stromal_test_ind + ['Other'] * other_test_ind

print("Train and test set up")

#-------------------TRANSFORMATIONS-------------------

def apply_blur(images, size):
    size = int(size)
    blurred = []
    
    for img in images:
        pil_img = Image.fromarray(img)
        blur = pil_img.filter(ImageFilter.GaussianBlur(radius=size))
        blurred.append(blur)
    return np.array(blurred)

def apply_contrast(images, factor):
    contrasted = []
    
    for img in images:
        pil_img = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(pil_img)
        adjusted = enhancer.enhance(factor)
        contrasted.append(adjusted)
    return np.array(contrasted)

def apply_greyscale(images):
    greyscale = []
    
    for img in images:
        pil_img = Image.fromarray(img)
        grey = pil_img.convert("L")
        grey_3ch = grey.convert("RGB")
        greyscale.append(grey_3ch)
    return np.array(greyscale)

Xmat_train_original = Xmat_train
Xmat_train_greyscale = apply_greyscale(Xmat_train)
print("greyscale augmentation complete")
Xmat_train_contrast_15 = apply_contrast(Xmat_train, 1.5)
print("contrast 1.5 augmentation complete")
Xmat_train_contrast_05 = apply_contrast(Xmat_train, 0.5)
print("contrast 0.5 augmentation complete")
Xmat_train_blur_100 = apply_blur(Xmat_train, 100)
print("blur 100 augmentation complete")
Xmat_train_blur_224 = apply_blur(Xmat_train, 224)
print("blur 224 augmentation complete")
Xmat_train_blur_5 = apply_blur(Xmat_train, 5)
print("blur 5 augmentation complete")

print("Augmented test train complete")

#---------FLATTEN--------
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Flatten images for XGBoost (224x224x3 -> 150528 features)
def flatten_images(images):
  return images.reshape(images.shape[0], -1)
X_train_flat = flatten_images(Xmat_train_original)
X_test_flat = flatten_images(Xmat_test)

# For augmented data
X_train_blur5_flat = flatten_images(Xmat_train_blur_5)
X_train_blur100_flat = flatten_images(Xmat_train_blur_100)
X_train_blur224_flat = flatten_images(Xmat_train_blur_224)
X_train_greyscale_flat = flatten_images(Xmat_train_greyscale)
X_train_contrast05_flat = flatten_images(Xmat_train_contrast_05)
X_train_contrast15_flat = flatten_images(Xmat_train_contrast_15)

print("data flattened")

#-------TRAINING BASE---------
print("initialise base")
xgb_model = xg.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
    )

base_start = time.time()

# Train model
xgb_model.fit(X_train_flat, y_train_enc)

base_end = time.time()
base_time = base_end - base_start
print(f"Base training time: {base_time:.2%}")

# Evaluate
y_pred = xgb_model.predict(X_test_flat)

accuracy = accuracy_score(y_test_enc, y_pred)
f1 = f1_score(y_test_enc, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test_enc, y_pred)

print(f"Original Accuracy: {accuracy:.2%}")
print(f"Original F1: {f1:.2%}")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'], yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('base.png')
#plt.show()

#-------TRAINING BLUR 5x5---------
print("initialise blur 5")
xgb_model_blur5 = xg.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
    )

blur5_start = time.time()

xgb_model_blur5.fit(X_train_blur5_flat, y_train_enc)

blur5_end = time.time()
blur5_time = blur5_end - blur5_start
print(f"Blur 5 training time: {blur5_time:.2%}")

# Evaluate
y_pred_blur5 = xgb_model_blur5.predict(X_test_flat)

accuracy_blur5 = accuracy_score(y_test_enc, y_pred_blur5)
f1_blur5 = f1_score(y_test_enc, y_pred_blur5, average='weighted')
conf_matrix_blur5 = confusion_matrix(y_test_enc, y_pred_blur5)

print(f"Blur 5 Accuracy: {accuracy_blur5:.2%}")
print(f"Blur 5 F1: {f1_blur5:.2%}")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_blur5, annot=True, fmt='g', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'], yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('blur5.png')
#plt.show()

#-------TRAINING BLUR 100x100---------

print("initialise blur 100")
xgb_model_blur100 = xg.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
    )

blur100_start = time.time()

xgb_model_blur100.fit(X_train_blur100_flat, y_train_enc)

blur100_end = time.time()
blur100_time = blur100_end - blur100_start 
print(f"Blur 100 training time: {blur100_time:.2%}")

# Evaluate
y_pred_blur100 = xgb_model_blur100.predict(X_test_flat)

accuracy_blur100 = accuracy_score(y_test_enc, y_pred_blur100)
f1_blur100 = f1_score(y_test_enc, y_pred_blur100, average='weighted')
conf_matrix_blur100 = confusion_matrix(y_test_enc, y_pred_blur100)

print(f"Blur 100 Accuracy: {accuracy_blur100:.2%}")
print(f"Blur F1: {f1_blur100:.2%}")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_blur100, annot=True, fmt='g', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'], yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('blur100.png')
#plt.show()

#-------TRAINING BLUR 224x224---------

print("initialise blur 224")
xgb_model_blur224 = xg.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
    )

blur224_start = time.time()

xgb_model_blur224.fit(X_train_blur224_flat, y_train_enc)

blur224_end = time.time()
blur224_time =  blur224_end - blur224_start
print(f"Blur 224 training time: {blur224_time:.2%}")

# Evaluate
y_pred_blur224 = xgb_model_blur224.predict(X_test_flat)

accuracy_blur224 = accuracy_score(y_test_enc, y_pred_blur224)
f1_blur224 = f1_score(y_test_enc, y_pred_blur224, average='weighted')
conf_matrix_blur224 = confusion_matrix(y_test_enc, y_pred_blur224)

print(f"Blur 224 Accuracy: {accuracy_blur224:.2%}")
print(f"Blur 224 F1: {f1_blur224:.2%}")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_blur224, annot=True, fmt='g', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'], yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('blur224.png')
#plt.show()

#-------TRAINING CONTRAST 0.5---------

print("initialise contrast 0.5")
xgb_model_contrast05 = xg.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
    )

contrast05_start = time.time()

xgb_model_contrast05.fit(X_train_contrast05_flat, y_train_enc)

contrast05_end = time.time()
contrast05_time =  contrast05_end - contrast05_start
print(f"Contrast 0.5 training time: {contrast05_time:.2%}")

# Evaluate
y_pred_contrast05 = xgb_model_contrast05.predict(X_test_flat)

accuracy_contrast05 = accuracy_score(y_test_enc, y_pred_contrast05)
f1_contrast05 = f1_score(y_test_enc, y_pred_contrast05, average='weighted')
conf_matrix_contrast05 = confusion_matrix(y_test_enc, y_pred_contrast05)

print(f"Contrast 0.5 Accuracy: {accuracy_contrast05:.2%}")
print(f"Contrast 0.5 F1: {accuracy_contrast05:.2%}")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_contrast05, annot=True, fmt='g', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'], yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('contrast05.png')
#plt.show()

#-------TRAINING CONTRAST 1.5---------

print("initialise contrast 1.5")
xgb_model_contrast15 = xg.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
    )

contrast15_start = time.time()

xgb_model_contrast15.fit(X_train_contrast15_flat, y_train_enc)

contrast15_end = time.time()
contrast15_time =  contrast15_end - contrast15_start
print(f"Contrast 1.5 training time: {contrast15_time:.2%}")

# Evaluate
y_pred_contrast15 = xgb_model_contrast15.predict(X_test_flat)

accuracy_contrast15 = accuracy_score(y_test_enc, y_pred_contrast15)
f1_contrast15 = f1_score(y_test_enc, y_pred_contrast15, average='weighted')
conf_matrix_contrast15 = confusion_matrix(y_test_enc, y_pred_contrast15)

print(f"Contrast 1.5 Accuracy: {accuracy_contrast15:.2%}")
print(f"Contrast 1.5 F1: {accuracy_contrast15:.2%}")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_contrast15, annot=True, fmt='g', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'], yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('contrast15.png')
#plt.show()

#-------TRAINING GREYSCALE---------
print("initialise greyscale")
xgb_model_greyscale = xg.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
    )

greyscale_start = time.time()

xgb_model_greyscale.fit(X_train_greyscale_flat, y_train_enc)

greyscale_end = time.time()
greyscale_time =  greyscale_end - greyscale_start
print(f"Greyscale training time: {greyscale_time:.2%}")

# Evaluate
y_pred_greyscale = xgb_model_greyscale.predict(X_test_flat)

accuracy_greyscale = accuracy_score(y_test_enc, y_pred_greyscale)
f1_greyscale = f1_score(y_test_enc, y_pred_greyscale, average='weighted')
conf_matrix_greyscale = confusion_matrix(y_test_enc, y_pred_greyscale)

print(f"Greyscale Accuracy: {accuracy_greyscale:.2%}")
print(f"Greyscale F1: {f1_greyscale:.2%}")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_greyscale, annot=True, fmt='g', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'], yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()