from PIL import Image, ImageFilter
#import tensorflow as tf
#from tensorflow.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt
from pathlib import Path
import random
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
import cv2

import shiny_data
import shiny_resnet50

batch_size = 64
num_epochs = 20

# Blur 100 augmented data
train_dataset, val_dataset, test_dataset = shiny_data.get_blurred_100()
blur100_losses, blur100_accuracies = shiny_resnet50.train_resnet50(train_dataset, val_dataset, batch_size, num_epochs, augmentation="blur100")
blur100_accuracy, blur100_f1, blur100_precision, blur100_recall = shiny_resnet50.eval_resnet50(test_dataset, batch_size, augmentation="blur100")

shiny_resnet50.graph_loss_accuracy(blur100_losses, blur100_accuracies, num_epochs)

print(f"blur100 Accuracy: {blur100_accuracy:.2f}%")
print(f"blur100 F1 Score: {blur100_f1:.2f}")
print(f"blur100 Precision: {blur100_precision:.2f}")
print(f"blur100 Recall: {blur100_recall:.2f}")

with open('resnet50_metrics.txt', 'a') as f:
    f.write(f"blur100 Accuracy: {blur100_accuracy:.2f}%\n")
    f.write(f"blur100 F1 Score: {blur100_f1:.2f}\n")
    f.write(f"blur100 Precision: {blur100_precision:.2f}\n")
    f.write(f"blur100 Recall: {blur100_recall:.2f}\n")
    f.write("\n")

# Blur 150 augmented data
train_dataset, val_dataset, test_dataset = shiny_data.get_blurred_150()
blur150_losses, blur150_accuracies = shiny_resnet50.train_resnet50(train_dataset, val_dataset, batch_size, num_epochs, augmentation="blur150")
blur150_accuracy, blur150_f1, blur150_precision, blur150_recall = shiny_resnet50.eval_resnet50(test_dataset, batch_size, augmentation="blur150")

shiny_resnet50.graph_loss_accuracy(blur150_losses, blur150_accuracies, num_epochs)

print(f"blur150 Accuracy: {blur150_accuracy:.2f}%")
print(f"blur150 F1 Score: {blur150_f1:.2f}")
print(f"blur150 Precision: {blur150_precision:.2f}")
print(f"blur150 Recall: {blur150_recall:.2f}")

with open('resnet50_metrics.txt', 'a') as f:
    f.write(f"blur150 Accuracy: {blur150_accuracy:.2f}%\n")
    f.write(f"blur150 F1 Score: {blur150_f1:.2f}\n")
    f.write(f"blur150 Precision: {blur150_precision:.2f}\n")
    f.write(f"blur150 Recall: {blur150_recall:.2f}\n")
    f.write("\n")

# Blur 200 augmented data
train_dataset, val_dataset, test_dataset = shiny_data.get_blurred_200()
blur200_losses, blur200_accuracies = shiny_resnet50.train_resnet50(train_dataset, val_dataset, batch_size, num_epochs, augmentation="blur200")
blur200_accuracy, blur200_f1, blur200_precision, blur200_recall = shiny_resnet50.eval_resnet50(test_dataset, batch_size, augmentation="blur200")

shiny_resnet50.graph_loss_accuracy(blur200_losses, blur200_accuracies, num_epochs)

print(f"blur200 Accuracy: {blur200_accuracy:.2f}%")
print(f"blur200 F1 Score: {blur200_f1:.2f}")
print(f"blur200 Precision: {blur200_precision:.2f}")
print(f"blur200 Recall: {blur200_recall:.2f}")

with open('resnet50_metrics.txt', 'a') as f:
    f.write(f"blur200 Accuracy: {blur200_accuracy:.2f}%\n")
    f.write(f"blur200 F1 Score: {blur200_f1:.2f}\n")
    f.write(f"blur200 Precision: {blur200_precision:.2f}\n")
    f.write(f"blur200 Recall: {blur200_recall:.2f}\n")
    f.write("\n")

# Blur full augmented data
train_dataset, val_dataset, test_dataset = shiny_data.get_blurred_full()
blurfull_losses, blurfull_accuracies = shiny_resnet50.train_resnet50(train_dataset, val_dataset, batch_size, num_epochs, augmentation="blurfull")
blurfull_accuracy, blurfull_f1, blurfull_precision, blurfull_recall = shiny_resnet50.eval_resnet50(test_dataset, batch_size, augmentation="blurfull")

shiny_resnet50.graph_loss_accuracy(blurfull_losses, blurfull_accuracies, num_epochs)

print(f"blur full Accuracy: {blurfull_accuracy:.2f}%")
print(f"blur full F1 Score: {blurfull_f1:.2f}")
print(f"blur full Precision: {blurfull_precision:.2f}")
print(f"blur full Recall: {blurfull_recall:.2f}")

with open('resnet50_metrics.txt', 'a') as f:
    f.write(f"blur full Accuracy: {blurfull_accuracy:.2f}%\n")
    f.write(f"blur full F1 Score: {blurfull_f1:.2f}\n")
    f.write(f"blur full Precision: {blurfull_precision:.2f}\n")
    f.write(f"blur full Recall: {blurfull_recall:.2f}\n")
    f.write("\n")