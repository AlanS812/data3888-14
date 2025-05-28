# data3888-14

# Project Overview

This project was developed as part of a University of Sydney DATA3888 capstone.
We trained and evaluated multiple models (Random Forest, XGBoost, CNN, ResNet50) on images extracted from H&E-stained breast cancer tissue slides.

## Dataset

Images were cropped from annotated H&E-stained breast tissue slides.
Each image was labeled as:

	•	Tumour
	•	Immune
	•	Stromal
	•	Other

A representative slide is shown below:

![Original Slide](images/welcome_image.png)

## Models Used
	•	Random Forest with raw pixels, HOG, and PCA features
	•	XGBoost (via PCA)
	•	CNN (custom)
	•	ResNet50 (pretrained, fine-tuned)

Each model was evaluated on its:
	•	Accuracy, F1, Precision, Recall (macro + per-class)
	•	Robustness under increasing blur and noise
	•	Confidence and stability across predictions


