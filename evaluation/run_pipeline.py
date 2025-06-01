 import os
import subprocess

# === Paths
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(EVAL_DIR, ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
METRICS_DIR = os.path.join(ROOT_DIR, "metrics")
MERGE_NOTEBOOK = os.path.join(EVAL_DIR, "merge_metrics.ipynb")
MERGE_SCRIPT = MERGE_NOTEBOOK.replace(".ipynb", ".py")
EVAL_SCRIPT = os.path.join(EVAL_DIR, "evaluation.py")

# === [1/4] Convert random_forest.ipynb to .py script
print("=== [1/4] Converting random_forest.ipynb to script ===")
subprocess.run(
    ["jupyter", "nbconvert", "--to", "script", "--output", "random_forest", "random_forest.ipynb"],
    cwd=MODELS_DIR,
    check=True
)

# === [2/4] Run RF model script only if output file is missing
rf_metrics_path = os.path.join(METRICS_DIR, "rf_augmented_metrics.csv")
if not os.path.exists(rf_metrics_path):
    print("=== [2/4] Running random_forest.py script ===")
    subprocess.run(["python3", "random_forest.py"], cwd=MODELS_DIR, check=True)
else:
    print("Skipping RF model run – rf_augmented_metrics.csv already exists.")

# === [3/4] Run evaluation.py only if model outputs are missing
xgb_path = os.path.join(METRICS_DIR, "xgboost_augmented_metrics.csv")
cnn_path = os.path.join(METRICS_DIR, "cnn_original_augmented_metrics.csv")
resnet_path = os.path.join(METRICS_DIR, "resnet50_augmented_metrics.csv")

if not all(os.path.exists(p) for p in [xgb_path, cnn_path, resnet_path]):
    print("=== [3/4] Running evaluation script for CNN, ResNet, XGBoost ===")
    subprocess.run(["python3", EVAL_SCRIPT], check=True)
else:
    print("Skipping evaluation – all model metric files already exist.")

# === [4/4] Convert merge_metrics.ipynb to .py if not already done
if not os.path.exists(MERGE_SCRIPT):
    print("Converting merge_metrics.ipynb to .py...")
    subprocess.run(
        ["jupyter", "nbconvert", "--to", "script", "--output", "merge_metrics", MERGE_NOTEBOOK],
        check=True
    )

# === Run merge_metrics script
print("=== [4/4] Merging all metrics into final CSVs ===")
subprocess.run(["python3", MERGE_SCRIPT], check=True)

print("All steps complete. Check the 'metrics/' folder for final combined CSVs.")