import os
import subprocess

# paths
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(EVAL_DIR, ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
METRICS_DIR = os.path.join(ROOT_DIR, "metrics")
MERGE_NOTEBOOK = os.path.join(EVAL_DIR, "merge_metrics.ipynb")
MERGE_SCRIPT = MERGE_NOTEBOOK.replace(".ipynb", ".py")
EVAL_SCRIPT = os.path.join(EVAL_DIR, "evaluation.py")

# rf
print("=== [1/4] Converting RF notebooks to scripts ===")
subprocess.run(["jupyter", "nbconvert", "--to", "script", "rf_models.ipynb"], cwd=MODELS_DIR)
subprocess.run(["jupyter", "nbconvert", "--to", "script", "rf_pca_model.ipynb"], cwd=MODELS_DIR)

print("=== [2/4] Running RF model scripts ===")
subprocess.run(["python3", "rf_models.py"], cwd=MODELS_DIR)
subprocess.run(["python3", "rf_pca_model.py"], cwd=MODELS_DIR)

# evaluation.py
print("=== [3/4] Running evaluation script for CNN, ResNet, XGBoost ===")
subprocess.run(["python3", EVAL_SCRIPT])

# making merge_metrics into a .py file
if not os.path.exists(MERGE_SCRIPT):
    print("Converting merge_metrics.ipynb to .py...")
    subprocess.run(["jupyter", "nbconvert", "--to", "script", MERGE_NOTEBOOK])

# merging the cvs using this
print("=== [4/4] Merging all metrics into combined_app_metrics.csv and combined_report_metrics.csv ===")
subprocess.run(["python3", MERGE_SCRIPT])

print("All steps complete. Check the 'metrics/' folder for final combined CSVs.")