from PIL import Image, ImageFilter, ImageEnhance
#from tensorflow.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt
from pathlib import Path
import random
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
import cv2
from skimage import util


#def get_image_paths(base_path="data/100"):

def get_image_paths(base_path=None): # pipeline addition
    
    if base_path is None:
        # This navigates from evaluation/ → ../data/100
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', '100'))

    # Get tumour file paths and shuffle
    tumour_files = []
    tumour_dirs = [
        "Invasive_Tumor",
        "Prolif_Invasive_Tumor",
        "T_Cell_and_Tumor_Hybrid"
    ]

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

    # Get stromal file paths and shuffle
    stromal_files = []
    stromal_dirs = [
        "Stromal", 
        "Stromal_and_T_Cell_Hybrid", 
        "Perivascular-Like"
    ]

    # Get other file paths and shuffle
    other_files = []
    other_dirs = [
        "Endothelial",
        "Myoepi_ACTA2+", 
        "Myoepi_KRT15+", 
        "DCIS_1", 
        "DCIS_2", 
    ]

    # Get the file paths for each category
    for dir_name in tumour_dirs:
        dir_path = os.path.join(base_path, dir_name)
        #print("[DEBUG] Scanning", dir_path)
        if os.path.isdir(dir_path):
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
            tumour_files.extend(files)
    
    for dir_name in immune_dirs:
        dir_path = os.path.join(base_path, dir_name)
        #print("[DEBUG] Scanning", dir_path)
        if os.path.isdir(dir_path):
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
            immune_files.extend(files)

    for dir_name in stromal_dirs:
        dir_path = os.path.join(base_path, dir_name)
        #print("[DEBUG] Scanning", dir_path)
        if os.path.isdir(dir_path):
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
            stromal_files.extend(files)

    for dir_name in other_dirs:
        dir_path = os.path.join(base_path, dir_name)
        #print("[DEBUG] Scanning", dir_path)
        if os.path.isdir(dir_path):
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
            other_files.extend(files)

    return [tumour_files, immune_files, stromal_files, other_files]

def load_resize(img_path, size=(224,224)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(size)
    return np.array(img)

#def label_and_split(images, label, training_size, val_size, testing_size):
#    random.shuffle(images)
#    return (
#        list(zip(images[:training_size], [label]*training_size)),
#        list(zip(images[training_size:training_size+val_size], [label]*val_size)),
#        list(zip(images[training_size+val_size:training_size+val_size+testing_size], [label]*testing_size)),
#    )

def label_and_split(images, label, training_size=2500, val_size=500, testing_size=1000):
    random.shuffle(images)
    return (
        list(zip(images[:training_size], [label]*training_size)),
        list(zip(images[training_size:training_size+val_size], [label]*val_size)),
        list(zip(images[training_size+val_size:training_size+val_size+testing_size], [label]*testing_size)),
    )

def load_split_images(training_size=2500, val_size=500, testing_size=1000):
    random.seed(3888)
    np.random.seed(3888)

    total_size = training_size + val_size + testing_size * 3  # Adjust total size for 3 test sets

    #tumour_files, immune_files, stromal_files, other_files = get_image_paths("data/original_data")
    tumour_files, immune_files, stromal_files, other_files = get_image_paths() # pipeline addition
    print("Total image counts:")
    print("  Tumour:", len(tumour_files))
    print("  Immune:", len(immune_files))
    print("  Stromal:", len(stromal_files))
    print("  Other:", len(other_files))

    tumour_imgs = [load_resize(img_path) for img_path in tumour_files[:total_size]]
    immune_imgs = [load_resize(img_path) for img_path in immune_files[:total_size]]
    stromal_imgs = [load_resize(img_path) for img_path in stromal_files[:total_size]]
    other_imgs = [load_resize(img_path) for img_path in other_files[:total_size]]

    t_train, t_val, t_test_all = label_and_split(tumour_imgs, 'Tumour', training_size, val_size, testing_size * 3)
    i_train, i_val, i_test_all = label_and_split(immune_imgs, 'Immune', training_size, val_size, testing_size * 3)
    s_train, s_val, s_test_all = label_and_split(stromal_imgs, 'Stromal', training_size, val_size, testing_size * 3)
    o_train, o_val, o_test_all = label_and_split(other_imgs, 'Other', training_size, val_size, testing_size * 3)

    print("After label_and_split:")
    print("  Tumour split:", len(t_train), len(t_val), len(t_test_all))
    print("  Immune split:", len(i_train), len(i_val), len(i_test_all))
    print("  Stromal split:", len(s_train), len(s_val), len(s_test_all))
    print("  Other split:", len(o_train), len(o_val), len(o_test_all))

    training_data = t_train + i_train + s_train + o_train
    val_data = t_val + i_val + s_val + o_val

    # Split combined test data into 3 equal test sets
    t_tests = [t_test_all[i*testing_size:(i+1)*testing_size] for i in range(3)]
    i_tests = [i_test_all[i*testing_size:(i+1)*testing_size] for i in range(3)]
    s_tests = [s_test_all[i*testing_size:(i+1)*testing_size] for i in range(3)]
    o_tests = [o_test_all[i*testing_size:(i+1)*testing_size] for i in range(3)]

    testing_sets = []
    for i in range(3):
        test_set = t_tests[i] + i_tests[i] + s_tests[i] + o_tests[i]
        random.shuffle(test_set)
        testing_sets.append(test_set)

    random.shuffle(training_data)
    random.shuffle(val_data)

    Xmat_train, y_train = zip(*training_data)
    Xmat_val, y_val = zip(*val_data)

    Xmat_train = np.stack(Xmat_train)
    Xmat_val = np.stack(Xmat_val)

    Xmat_tests = []
    y_tests_enc = []
    le = LabelEncoder()
    
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    for test_set in testing_sets:
        X_test, y_test = zip(*test_set)
        Xmat_tests.append(np.stack(X_test))
        y_tests_enc.append(le.transform(y_test))

    print(list(zip(le.classes_, le.transform(le.classes_))))

    return Xmat_train, Xmat_val, Xmat_tests[0], Xmat_tests[1], Xmat_tests[2], y_train_enc, y_val_enc, y_tests_enc[0], y_tests_enc[1], y_tests_enc[2]


class NumpyImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = Image.fromarray((image * 255).astype('uint8'))  # Convert to PIL Image

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

def resize_images(images, size):
    """
    Resize images to the specified size
    """
    resized_images = []
    for img in images:
        pil_img = Image.fromarray(img)
        resized_img = pil_img.resize(size, Image.Resampling.LANCZOS)
        resized_images.append(np.array(resized_img))
    return np.array(resized_images)

def transform_datasets(Xmat_train, Xmat_val, Xmat_test1, Xmat_test2, Xmat_test3, y_train_enc, y_val_enc, y_test_enc1, y_test_enc2, y_test_enc3):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_dataset = NumpyImageDataset(Xmat_train, y_train_enc, transform=transform)
    val_dataset = NumpyImageDataset(Xmat_val, y_val_enc, transform=transform)
    test_dataset1 = NumpyImageDataset(Xmat_test1, y_test_enc1, transform=transform)
    test_dataset2 = NumpyImageDataset(Xmat_test2, y_test_enc2, transform=transform)
    test_dataset3 = NumpyImageDataset(Xmat_test3, y_test_enc3, transform=transform)

    return train_dataset, val_dataset, test_dataset1, test_dataset2, test_dataset3

# Augmentations

def apply_blur(images, size):
    size = int(size)
    blurred = []
    if size > 0:
        for img in images:
            pil_img = Image.fromarray(img)
            blur = pil_img.filter(ImageFilter.GaussianBlur(radius=size))
            blurred.append(np.array(blur))
        return np.array(blurred).astype(np.uint8)  # ensure uint8 output
    else:
        return images

def apply_noise(images, mean=0, std=10):
    noisy_images = []
    if std > 0:
        for img in images:
            np.random.seed(3888)
            noise = np.random.normal(mean, std, img.shape)
            noisy_img = img + noise
            noisy_img = np.clip(noisy_img, 0, 255)
            noisy_images.append(noisy_img.astype(np.uint8))  # ensure uint8 output
        return np.array(noisy_images)
    else:
        return images


# Create datasets
def get_original(training_size=2500, val_size=500, testing_size=1000):
    """
    Load the original dataset
    """
    Xmat_train, Xmat_val, Xmat_test1, Xmat_test2, Xmat_test3, y_train_enc, y_val_enc, y_test_enc1, y_test_enc2, y_test_enc3 = load_split_images(training_size, val_size, testing_size)
    train_dataset, val_dataset, test_dataset1, test_dataset2, test_dataset3 = transform_datasets(Xmat_train, Xmat_val, Xmat_test1, Xmat_test2, Xmat_test3, y_train_enc, y_val_enc, y_test_enc1, y_test_enc2, y_test_enc3)
    return train_dataset, val_dataset, test_dataset1, test_dataset2, test_dataset3
