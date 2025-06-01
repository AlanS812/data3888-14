# Project Overview

This project was developed as part of a University of Sydney DATA3888 capstone. We trained and evaluated multiple models (Random Forest, XGBoost, CNN, ResNet50) on images extracted from an annotated H&E-stained breast cancer tissue slide, seen below. The goal was to evaluate which model architecture performs most reliably when classifying histological cell types under realistic image degradation, helping inform model selection in digital pathology workflows.

![Original Slide](slide_image.png)

Each model was evaluated on:
- Accuracy, F1, Precision, Recall (macro and per class)
- Robustness under increasing blur and noise augmentations
- Confidence and class-level prediction stability

For full methodology, experimental setup, and results, see the final ***[report link here].***

## Dataset

Images were cropped from a high-resolution, annotated breast tissue slide sourced from ***[GEO: GSE243280, turn into link]***. The dataset contains ~175,000 labelled cells.

Cell images were grouped into Tumour, Immune, Stromal, and Other, based on biological function in breast cancer diagnosis and progression. Unlabelled cells were excluded from the study.

A balanced subset of 20,000 images (5,000 per class) was used to ensure equal class representation. Final splits:
- 75% training
- 10% validation
- 15% testing (×3 test sets)

# Reproducing Results

This repo includes an evaluation pipeline for image classification models. The script `run_pipeline.py` (located in the `evaluation/` folder) automates the evaluation process:

- Converts necessary Jupyter notebooks to `.py` scripts.
- Runs evaluation for RF, CNN, ResNet, and XGBoost models, **only if** their output files (`*_augmented_metrics.csv`) are missing.
- Merges all model metrics into two final CSVs:  
  - `combined_report_metrics.csv` (for the report)  
  - `combined_app_metrics.csv` (for the Shiny app)

### To run the pipeline:

```bash
python3 evaluation/run_pipeline.py
```

> **Note:** This script does **not retrain models from scratch** — it assumes models like ResNet50, CNN, and XGBoost are already trained and saved (e.g., `.pt`, `.h5`, `.json`). See below for details on manual retraining.

## Manual Retraining

To retrain models from scratch, run the following scripts manually. Results are saved in the `models/` directory.

### Random Forest
```bash
jupyter nbconvert --to script --output random_forest models/random_forest.ipynb
python3 models/random_forest.py
```
### CNN
```bash
models/cnn.ipynb
```
Saves trained model to `models/cnn_original.h5`

### ResNet50
```bash
python3 models/resnet50.py
```

This will use `resnet50_models/` to store intermediate outputs and save the final model as `resnet50_original_model.pt`

### XGBoost
```bash
python3 models/xgboost.py
```
Model will be saved as `xgboost.json`