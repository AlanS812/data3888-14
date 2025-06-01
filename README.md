# Project Overview

This project was developed as part of a University of Sydney DATA3888 capstone. We trained and evaluated multiple models (Random Forest, XGBoost, CNN, ResNet50) on images extracted from an annotated H&E-stained breast cancer tissue slide, seen below. The goal was to evaluate which model architecture performs most reliably when classifying histological cell types under realistic image degradation, helping inform model selection in digital pathology workflows.

![Original Slide](slide_image.png)

Each model was evaluated on:
- Accuracy, F1, Precision, Recall (macro and per class)
- Robustness under increasing blur and noise augmentations
- Confidence and class-level prediction stability

For full methodology, experimental setup, and results, see the [report folder](report/).

## Dataset

Images were cropped from a high-resolution, annotated breast tissue slide sourced from ***[GEO: GSE243280](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243280)***. The dataset contains ~175,000 labelled cells.

Cell images were grouped into Tumour, Immune, Stromal, and Other, based on biological function in breast cancer diagnosis and progression. Unlabelled cells were excluded from the study.

A balanced subset of 20,000 images (5,000 per class) was used to ensure equal class representation. Final splits:
- 75% training
- 10% validation
- 15% testing (×3 test sets)

# Reproducing Results

**Note**: Running model training files is time consuming. If you would like to skip to evaluation, download the two `*.joblib` files from [here](https://drive.google.com/drive/folders/1NigrmCDCaJrOYtNyPjFiV0rYguptSdPH?usp=sharing) and place them in `data3888-14/models`. They exceeded Github storage requirements so need to be downloaded manually.

This repo includes an evaluation pipeline for image classification models. The script `run_pipeline.py` (located in the `evaluation/` folder) automates the evaluation process:

- Converts necessary Jupyter notebooks to `.py` scripts.
- Runs evaluation for RF, CNN, ResNet, and XGBoost models, **only if** their output files (`*_augmented_metrics.csv`) are missing.
- Merges all model metrics into two final CSVs:  
  - `combined_report_metrics.csv` (for the report)  
  - `combined_app_metrics.csv` (for the Shiny app)

## Getting Started

To run the pipeline, app, or model training scripts, you must first download or clone the repository locally.

```bash
git clone https://github.com/AlanS812/data3888-14.git
cd data3888-14
```

### To run the model evaluation pipeline:

```bash
python3 evaluation/run_pipeline.py
```

> **Note:** This script does **not retrain models from scratch** — it assumes models like ResNet50, CNN, and XGBoost are already trained and saved (e.g., `.pt`, `.h5`, `.json`). See below for details on manual retraining.

## Manual Retraining

To retrain models from scratch, run the following scripts manually. Results are saved in the `models/` directory.

### Random Forest (`random_forest.ipynb`)
```bash
jupyter nbconvert --to script --output random_forest models/random_forest.ipynb
python3 models/random_forest.py
```
### CNN (`cnn_original.h5`)
```bash
models/cnn.ipynb
```
### ResNet50 (`resnet50_original_model.pt`)
```bash
python3 models/resnet50.py
```

### XGBoost (`models/xgboost.json`)
```bash
python3 models/xgboost.py
```

## Running the Shiny App (Python Shiny)

```bash
cd app
shiny run --reload app.py
# or
python -m shiny run --reload app.py
```
Once running, the app will provide a local link like http://127.0.0.1:8000 — open that in your browser to explore the dashboard.
