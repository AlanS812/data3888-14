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


# Train model

def train_resnet50(train_dataset, val_dataset, batch_size=128, num_epochs=10, augmentation="original"):
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size)

    # Train model with no augmentations
    #tf.keras.backend.clear_session()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=False)

    # Freeze feature extractor
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer for 4 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)


    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    losses = []
    accuracies = []

    patience = 5  # number of epochs to wait for improvement
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Accuracy calculation
            _, predicted = outputs.max(1)  # Get class with highest score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'resnet50_models/{augmentation}_model.pt')  # save the best model
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping triggered!")
                early_stop = True
                break

    # Load and save the best model
    if not early_stop:
        print("Training completed without early stopping.")

    return losses, accuracies


# Function to evaluate model
def eval_resnet50(test_dataset, batch_size=128, augmentation="original"):
    test_loader = DataLoader(test_dataset, batch_size)

    model = models.resnet50(pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reset model parameters to load in existing weights
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)

    model.load_state_dict(torch.load(f'resnet50_models/{augmentation}_model.pt', map_location=device))
    model = model.to(device)

    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No need to calculate gradients during evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Pick class with highest probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate evaluation metrics
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    print(f"Original Accuracy: {accuracy:.2f}%")
    print(classification_report(all_labels, all_preds))

    return accuracy, f1, precision, recall


def graph_loss_accuracy(losses, accuracies, num_epochs):
    plt.figure(figsize=(10,5))
    plt.plot(range(1, num_epochs+1), losses, label='Loss')
    plt.plot(range(1, num_epochs+1), accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Loss and Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Original data
    num_epochs = 20
    batch_size = 128

    train_dataset, val_dataset, test_dataset = shiny_data.get_original()
    og_losses, og_accuracies = train_resnet50(train_dataset, val_dataset, batch_size, num_epochs, augmentation="original")
    og_accuracy, og_f1, og_precision, og_recall = eval_resnet50(test_dataset, batch_size, augmentation="original")

    graph_loss_accuracy(og_losses, og_accuracies, num_epochs)

    print(f"Original Accuracy: {og_accuracy:.2f}%")
    print(f"Original F1 Score: {og_f1:.2f}")
    print(f"Original Precision: {og_precision:.2f}")
    print(f"Original Recall: {og_recall:.2f}")

    # Blur 50 augmented data
    train_dataset, val_dataset, test_dataset = shiny_data.get_blurred_50()
    blur50_losses, blur50_accuracies = train_resnet50(train_dataset, val_dataset, batch_size, num_epochs, augmentation="blur50")
    blur50_accuracy, blur50_f1, blur50_precision, blur50_recall = eval_resnet50(test_dataset, batch_size, augmentation="blur50")

    graph_loss_accuracy(blur50_losses, blur50_accuracies, num_epochs)

    print(f"Blur 50 Accuracy: {blur50_accuracy:.2f}%")
    print(f"Blur 50 F1 Score: {blur50_f1:.2f}")
    print(f"Blur 50 Precision: {blur50_precision:.2f}")
    print(f"Blur 50 Recall: {blur50_recall:.2f}")

    # Blur 100 augmented data
    train_dataset, val_dataset, test_dataset = shiny_data.get_blurred_100()
    blur100_losses, blur100_accuracies = train_resnet50(train_dataset, val_dataset, batch_size, num_epochs, augmentation="blur100")
    blur100_accuracy, blur100_f1, blur100_precision, blur100_recall = eval_resnet50(test_dataset, batch_size, augmentation="blur100")

    graph_loss_accuracy(blur100_losses, blur100_accuracies, num_epochs)

    print(f"blur100 Accuracy: {blur100_accuracy:.2f}%")
    print(f"blur100 F1 Score: {blur100_f1:.2f}")
    print(f"blur100 Precision: {blur100_precision:.2f}")
    print(f"blur100 Recall: {blur100_recall:.2f}")

    # Blur 150 augmented data
    train_dataset, val_dataset, test_dataset = shiny_data.get_blurred_150()
    blur150_losses, blur150_accuracies = train_resnet50(train_dataset, val_dataset, batch_size, num_epochs, augmentation="blur50")
    blur150_accuracy, blur150_f1, blur150_precision, blur150_recall = eval_resnet50(test_dataset, batch_size, augmentation="blur50")

    graph_loss_accuracy(blur150_losses, blur150_accuracies, num_epochs)

    print(f"blur150 Accuracy: {blur150_accuracy:.2f}%")
    print(f"blur150 F1 Score: {blur150_f1:.2f}")
    print(f"blur150 Precision: {blur150_precision:.2f}")
    print(f"blur150 Recall: {blur150_recall:.2f}")

    # Blur 200 augmented data
    train_dataset, val_dataset, test_dataset = shiny_data.get_blurred_200()
    blur200_losses, blur200_accuracies = train_resnet50(train_dataset, val_dataset, batch_size, num_epochs, augmentation="blur50")
    blur200_accuracy, blur200_f1, blur200_precision, blur200_recall = eval_resnet50(test_dataset, batch_size, augmentation="blur50")

    graph_loss_accuracy(blur200_losses, blur200_accuracies, num_epochs)

    print(f"blur200 Accuracy: {blur200_accuracy:.2f}%")
    print(f"blur200 F1 Score: {blur200_f1:.2f}")
    print(f"blur200 Precision: {blur200_precision:.2f}")
    print(f"blur200 Recall: {blur200_recall:.2f}")

    # Blur full augmented data
    train_dataset, val_dataset, test_dataset = shiny_data.get_blurred_full()
    blurfull_losses, blurfull_accuracies = train_resnet50(train_dataset, val_dataset, batch_size, num_epochs, augmentation="blur50")
    blurfull_accuracy, blurfull_f1, blurfull_precision, blurfull_recall = eval_resnet50(test_dataset, batch_size, augmentation="blur50")

    graph_loss_accuracy(blurfull_losses, blurfull_accuracies, num_epochs)

    print(f"blur full Accuracy: {blurfull_accuracy:.2f}%")
    print(f"blur full F1 Score: {blurfull_f1:.2f}")
    print(f"blur full Precision: {blurfull_precision:.2f}")
    print(f"blur full Recall: {blurfull_recall:.2f}")

if __name__ == "__main__":
    main()
