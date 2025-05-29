# Project Overview

This project was developed as part of a University of Sydney DATA3888 capstone. We trained and evaluated multiple models (Random Forest, XGBoost, CNN, ResNet50) on images extracted an H&E-stained breast cancer tissue slide, seen below.

![Original Slide](slide_image.png)

Each model was evaluated on its:
- Accuracy, F1, Precision, Recall (macro + per-class)
- Robustness under increasing blur and noise
- Confidence and stability across predictions

## Dataset

Images were cropped from annotated H&E-stained breast tissue slide.
Cell types were grouped as being Tumour, Immune, Stromal, or Other, depending on their biological role in breast cancer progression.