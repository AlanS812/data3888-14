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


def get_image_paths(base_path="data/100"):
    """
    Load the training data
    """

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
        if os.path.isdir(dir_path):
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
            tumour_files.extend(files)
    
    for dir_name in immune_dirs:
        dir_path = os.path.join(base_path, dir_name)
        if os.path.isdir(dir_path):
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
            immune_files.extend(files)

    for dir_name in stromal_dirs:
        dir_path = os.path.join(base_path, dir_name)
        if os.path.isdir(dir_path):
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
            stromal_files.extend(files)

    for dir_name in other_dirs:
        dir_path = os.path.join(base_path, dir_name)
        if os.path.isdir(dir_path):
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
            other_files.extend(files)

    return [tumour_files, immune_files, stromal_files, other_files]

def load_resize(img_path, size=(224,224)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(size)
    return np.array(img)

def label_and_split(images, label, training_size, val_size, testing_size):
    random.shuffle(images)
    return (
        list(zip(images[:training_size], [label]*training_size)),
        list(zip(images[training_size:training_size+val_size], [label]*val_size)),
        list(zip(images[training_size+val_size:training_size+val_size+testing_size], [label]*testing_size)),
    )

def label_and_split(images, label, training_size=2500, val_size=500, testing_size=1000):
    random.shuffle(images)
    return (
        list(zip(images[:training_size], [label]*training_size)),
        list(zip(images[training_size:training_size+val_size], [label]*val_size)),
        list(zip(images[training_size+val_size:training_size+val_size+testing_size], [label]*testing_size)),
    )

def load_split_images(training_size=2500, val_size=500, testing_size=1000):
    """
    Split the data into training and testing sets
    """
    random.seed(3888)
    np.random.seed(3888)

    total_size = training_size + val_size + testing_size

    training_data = []
    val_data = []
    testing_data = []

    # Load the images
    tumour_files, immune_files, stromal_files, other_files = get_image_paths("data/100")
    tumour_imgs = [load_resize(img_path) for img_path in tumour_files[:total_size]]
    print("Tumour images loaded")
    immune_imgs = [load_resize(img_path) for img_path in immune_files[:total_size]]
    print("Immune images loaded")
    stromal_imgs = [load_resize(img_path) for img_path in stromal_files[:total_size]]
    print("Stromal images loaded")
    other_imgs = [load_resize(img_path) for img_path in other_files[:total_size]]
    print("Other images loaded")


    t_train, t_val, t_test = label_and_split(tumour_imgs, 'Tumour', training_size, val_size, testing_size)
    i_train, i_val, i_test = label_and_split(immune_imgs, 'Immune', training_size, val_size, testing_size)
    s_train, s_val, s_test = label_and_split(stromal_imgs, 'Stromal', training_size, val_size, testing_size)
    o_train, o_val, o_test = label_and_split(other_imgs, 'Other', training_size, val_size, testing_size)

    # Combine and shuffle
    training_data = t_train + i_train + s_train + o_train
    val_data = t_val + i_val + s_val + o_val
    testing_data = t_test + i_test + s_test + o_test

    random.shuffle(training_data)
    random.shuffle(val_data)
    random.shuffle(testing_data)

    # Unzip into X and y
    Xmat_train, y_train = zip(*training_data)
    Xmat_val, y_val = zip(*val_data)
    Xmat_test, y_test = zip(*testing_data)

    Xmat_train = np.stack(Xmat_train)
    Xmat_val = np.stack(Xmat_val)
    Xmat_test = np.stack(Xmat_test)

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    return Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc


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
        resized_img = pil_img.resize(size, Image.ANTIALIAS)
        resized_images.append(np.array(resized_img))
    return np.array(resized_images)

def transform_datasets(Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_dataset = NumpyImageDataset(Xmat_train, y_train_enc, transform=transform)
    val_dataset = NumpyImageDataset(Xmat_val, y_val_enc, transform=transform)
    test_dataset = NumpyImageDataset(Xmat_test, y_test_enc, transform=transform)

    return train_dataset, val_dataset, test_dataset

# Augmentations

def apply_blur(images, size):
    size = int(size)
    blurred = []
    
    for img in images:
        pil_img = Image.fromarray(img)
        blur = pil_img.filter(ImageFilter.GaussianBlur(radius=size))
        blurred.append(np.array(blur))
    return np.array(blurred)

def apply_full_image_blur(images):
    blurred = []
    
    for img in images:
        pil_img = Image.fromarray(img)
        width, height = pil_img.size
        # Use max dimension as blur radius to simulate full-image blur
        blur_radius = max(width, height)
        blurred_img = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        blurred.append(np.array(blurred_img))
    
    return np.array(blurred)

def apply_greyscale(images):
    greyscale = []
    for img in images:
        pil_img = Image.fromarray(img)
        grey = pil_img.convert("L")
        grey_3ch = grey.convert("RGB")
        greyscale.append(np.array(grey_3ch))
    return np.array(greyscale)

def adjust_contrast(images, factor=1.5):
    adjusted = []
    for img in images:
        pil_img = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(pil_img)
        contrasted = enhancer.enhance(factor)
        adjusted.append(np.array(contrasted))
    return np.array(adjusted)


# Create datasets
def get_original(training_size=2500, val_size=500, testing_size=1000):
    """
    Load the original dataset
    """
    Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc = load_split_images(training_size, val_size, testing_size)
    train_dataset, val_dataset, test_dataset = transform_datasets(Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc)
    return train_dataset, val_dataset, test_dataset

def get_blurred_1(training_size=2500, val_size=500, testing_size=1000):
    """
    Load the blurred dataset
    """
    Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc = load_split_images(training_size, val_size, testing_size)
    Xmat_train = apply_blur(Xmat_train, 1)
    
    train_dataset, val_dataset, test_dataset = transform_datasets(Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc)
    return train_dataset, val_dataset, test_dataset

def get_blurred_3(training_size=2500, val_size=500, testing_size=1000):
    """
    Load the blurred dataset
    """
    Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc = load_split_images(training_size, val_size, testing_size)
    Xmat_train = apply_blur(Xmat_train, 3)
    
    train_dataset, val_dataset, test_dataset = transform_datasets(Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc)
    return train_dataset, val_dataset, test_dataset

def get_blurred_5(training_size=2500, val_size=500, testing_size=1000):
    """
    Load the blurred dataset
    """
    Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc = load_split_images(training_size, val_size, testing_size)
    Xmat_train = apply_blur(Xmat_train, 5)
    
    train_dataset, val_dataset, test_dataset = transform_datasets(Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc)
    return train_dataset, val_dataset, test_dataset

def get_blurred_7(training_size=2500, val_size=500, testing_size=1000):
    """
    Load the blurred dataset
    """
    Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc = load_split_images(training_size, val_size, testing_size)
    Xmat_train = apply_blur(Xmat_train, 7)
    
    train_dataset, val_dataset, test_dataset = transform_datasets(Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc)
    return train_dataset, val_dataset, test_dataset

def get_blurred_10(training_size=2500, val_size=500, testing_size=1000):
    """
    Load the blurred dataset
    """
    Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc = load_split_images(training_size, val_size, testing_size)
    Xmat_train = apply_blur(Xmat_train, 10)
    
    train_dataset, val_dataset, test_dataset = transform_datasets(Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc)
    return train_dataset, val_dataset, test_dataset

def get_greyscale(training_size=2500, val_size=500, testing_size=1000):
    """
    Load the greyscale dataset
    """
    Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc = load_split_images(training_size, val_size, testing_size)
    Xmat_train = apply_greyscale(Xmat_train)
    
    train_dataset, val_dataset, test_dataset = transform_datasets(Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc)
    return train_dataset, val_dataset, test_dataset

def get_contrast_1_5(training_size=2500, val_size=500, testing_size=1000):
    """
    Load the contrast dataset
    """
    Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc = load_split_images(training_size, val_size, testing_size)
    Xmat_train = adjust_contrast(Xmat_train, 1.5)
    
    train_dataset, val_dataset, test_dataset = transform_datasets(Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc)
    return train_dataset, val_dataset, test_dataset

def get_contrast_0_5(training_size=2500, val_size=500, testing_size=1000):
    """
    Load the contrast dataset
    """
    Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc = load_split_images(training_size, val_size, testing_size)
    Xmat_train = adjust_contrast(Xmat_train, 0.5)
    
    train_dataset, val_dataset, test_dataset = transform_datasets(Xmat_train, Xmat_val, Xmat_test, y_train_enc, y_val_enc, y_test_enc)
    return train_dataset, val_dataset, test_dataset


