from shiny import App, ui, render, req
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import seaborn as sns
import shiny_data
import torch
from tensorflow.keras.models import load_model
import xgboost as xgb
import joblib
from torchvision import models
import torch.nn as nn
import evaluation

# Load once
pca = joblib.load("Base_pca.joblib")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("xgboost.json")
cnn_model = load_model("cnn_original.h5")

device = torch.device("cpu")
rn_model = models.resnet50(pretrained=False)
num_ftrs = rn_model.fc.in_features
rn_model.fc = nn.Linear(num_ftrs, 4)
rn_model.load_state_dict(torch.load("resnet50.pt", map_location=device))
rn_model = rn_model.to(device)
rn_model.eval()

rf_model = joblib.load("rf_pca_model.joblib")

# Prediction functions

def predict_rf(img, pca, model):
    blur_sizes = [0, 1, 3, 5, 7, 9, 19]
    noise_levels = [0, 1, 3, 5, 10, 20, 30]
    preds = np.empty((len(noise_levels), len(blur_sizes)), dtype=int)
    
    for i, noise in enumerate(noise_levels):
        for j, blur in enumerate(blur_sizes):
            aug = shiny_data.apply_noise(shiny_data.apply_blur(img[np.newaxis, ...], blur), std=noise)
            aug_flat = aug.reshape(1, -1)
            aug_pca = pca.transform(aug_flat)
            preds[i, j] = model.predict(aug_pca)[0]
    
    return preds, blur_sizes, noise_levels
  
def predict_cnn(img, model):
    blur_sizes = [0,1,3,5,7,9,19]
    noise_levels = [0,1,3,5,10,20,30]
    preds = np.empty((len(noise_levels), len(blur_sizes)), dtype=int)
    for i, noise in enumerate(noise_levels):
        for j, blur in enumerate(blur_sizes):
            aug = shiny_data.apply_noise(shiny_data.apply_blur(img[np.newaxis, ...], blur), std=noise)
            y_probs = model.predict(aug)
            preds[i, j] = np.argmax(y_probs, axis=1)[0]
    return preds, blur_sizes, noise_levels

def predict_xgb(img, pca, model):
    blur_sizes = [0,1,3,5,7,9,19]
    noise_levels = [0,1,3,5,10,20,30]
    preds = np.empty((len(noise_levels), len(blur_sizes)), dtype=int)
    for i, noise in enumerate(noise_levels):
        for j, blur in enumerate(blur_sizes):
            aug = shiny_data.apply_noise(shiny_data.apply_blur(img[np.newaxis, ...], blur), std=noise)
            aug_flat = aug.reshape(1, -1)
            aug_pca = pca.transform(aug_flat)
            preds[i, j] = model.predict(aug_pca)[0]
    return preds, blur_sizes, noise_levels

def predict_resnet(img, model):
    blur_sizes = [0,1,3,5,7,9,19]
    noise_levels = [0,1,3,5,10,20,30]
    img_tensor = torch.tensor(img).permute(2, 0, 1).float()
    preds = np.empty((len(noise_levels), len(blur_sizes)), dtype=int)
    with torch.no_grad():
        for i, noise in enumerate(noise_levels):
            for j, blur in enumerate(blur_sizes):
                aug = evaluation.apply_augmentations(img_tensor.unsqueeze(0), blur, noise).to(device)
                outputs = model(aug)
                preds[i, j] = torch.argmax(torch.softmax(outputs, dim=1), dim=1).item()
    return preds, blur_sizes, noise_levels

# Shiny justification (notes)
#Class imbalance and bias: In medical imaging, models often perform unevenly across classes. 
#Tumour cells may dominate, causing models to over-prioritize them at the expense of immune 
#or stromal types. Per-class metrics reveal these biases.

#Clinical interpretability: It’s critical to know not just if a model performs well overall, 
#but where it fails. An oncologist might care more about detecting immune cells than “other” 
#types, so per-class insights help prioritize model trust.

#Robustness to degradation: Your augmentations simulate real-world noise (blurring, 
#scanning artifacts). Some cell types might degrade in distinct ways — e.g., tumour 
#cells may remain detectable under blur, while immune cells vanish into noise. 
#Per-class degradation sensitivity matters.

#df = pd.read_csv("results.csv")
df = pd.read_csv("combined_results_100.csv")

# defining augmentations
blur_levels = [0, 1, 3, 5, 7, 9, 19] 
noise_levels = [0, 1, 3, 5, 10, 20, 30] 
model_colors = {
    #"RF (pixels)": "#1f77b4",
    "RF (PCA)": "#ff7f0e",
    "XGB (PCA)": "#d62728",
    "CNN": "#9467bd",
    "ResNet": "#8c564b"
}

class_colors = {
    "Immune": "#2ca02c",
    "Other": "#ff7f0e",
    "Stromal": "#1f77b4",
    "Tumour": "#d62728"
}

# defining groupings
tumour_labels = [
    "Invasive_Tumor.png", 
    "Prolif_Invasive_Tumor.png", 
    "T_Cell_and_Tumor_Hybrid.png"
]

immune_labels = [
    "CD4+_T_Cells.png", "CD8+_T_Cells.png", "B_Cells.png", "Mast_Cells.png",
    "Macrophages_1.png", "Macrophages_2.png", "LAMP3+_DCs.png", "IRF7+_DCs.png"
]

stromal_like_labels = [
    "Stromal.png", "Stromal_and_T_Cell_Hybrid.png", "Perivascular-Like.png"
]

non_stromal_other_labels = [
    "Endothelial.png", "Myoepi_ACTA2+.png", "Myoepi_KRT15+.png", 
    "DCIS_1.png", "DCIS_2.png"
]

app_ui = ui.page_navbar(
    # Move style inside the parentheses, but **after** all nav_panels
    ui.nav_panel(
        "Welcome",
        ui.div(
            ui.h2("Welcome to the Image Classification App"),
            ui.p("This app shows performance of different models trained on medical image data."),
            ui.p("You can select different image augmentations and compare model results.")
        )
    ),

      ui.nav_panel(
        "Model Results",
        ui.div(
    
            # === AUGMENTATION + IMAGE PANEL (unchanged) ===
            ui.div(
                # Augmentation selectors
                ui.div(
                  ui.card(
                    ui.h4("Example Cell Images"),
                    ui.p("Use the dropdowns to explore example images for cells in each group."),

                        ui.h4("Augmentations"),
                        ui.layout_columns(
                            ui.div(
                                ui.h5("Blur", style="font-size: 1rem; font-weight: 600;"),
                                ui.input_radio_buttons("blur", "", choices=blur_levels, inline=True)
                            ),
                            ui.div(
                                ui.h5("Noise", style="font-size: 1rem; font-weight: 600;"),
                                ui.input_radio_buttons("noise", "", choices=noise_levels, inline=True)
                            )
                        )
                    ),

                    #ui.card(
                       # ui.h4("Augmentation"),
                       # ui.input_radio_buttons("blur", "Blur Level", choices=blur_levels, inline=False),
                       # ui.input_radio_buttons("noise", "Noise Level", choices=noise_levels, inline=False)
                    #),
                    style="width: 25%; padding-right: 1rem;"
                ),
    
                # Image display cards
                ui.div(
                    *[
                        ui.card(
                            ui.div(
                                ui.h5(title),
                                ui.div(
                                    ui.output_image(image_id),
                                    style="width: 100%; max-height: 210px; overflow: hidden;"
                                ),
                                ui.input_select(
                                    select_id,
                                    "",
                                    choices={label: label.replace(".png", "").replace("_", " ") for label in label_list}
                                ),
                                style="display: flex; flex-direction: column; align-items: center; gap: 1rem; justify-content: flex-start;"
                            ),
                            style="flex: 1 1 22%; min-width: 200px;"
                        )
                        for title, image_id, select_id, label_list in [
                            ("Tumour Cells", "tumour_img", "tumour_cell", tumour_labels),
                            ("Immune Cells", "immune_img", "immune_cell", immune_labels),
                            ("Stromal Cells", "stromal_img", "stromal_cell", stromal_like_labels),
                            ("Other Cells", "nonstromal_img", "nonstromal_cell", non_stromal_other_labels),
                        ]
                    ],
                    style="width: 75%; display: flex; flex-wrap: nowrap; gap: 1rem; justify-content: space-between;"
                ),
                style="display: flex;"
            ),
    
            # === METRIC TOGGLE + DYNAMIC PLOT + PLACEHOLDER LINE GRAPH ===
            ui.div(
                ui.layout_columns(
                   ui.card(
                      ui.h4("Bar Plots"),
                      ui.input_radio_buttons(
                          "line_metric",
                          "Metric",
                          choices=["Accuracy", "Precision", "Recall", "F1 Score"],
                          selected="F1 Score",
                          inline=True
                      ),
                      ui.output_plot("selected_metric_plot"),  # Plot changes based on toggle
                      width=6
                    ),
                    ui.card(
                      ui.h4("Line Plot"),
                  
                        ui.input_radio_buttons(
                            "fixed_aug",
                            "Vary Across:",
                            choices=["Blur", "Noise"],
                            selected="Noise",
                            inline=True
                        ),
                
                      # 4. The dynamic line plot
                      ui.output_plot("line_metric_plot")
                  )
                ),
                style="margin-top: 0.5rem;"
            ),
    
            # === ADDITIONAL PLOTS BELOW ===
            #ui.layout_columns(
            #    ui.card(ui.h4("Average Confidence"), ui.output_plot("confidence_plot"), width=6),
            #    ui.card(ui.h4("Training and Validation"), ui.output_plot("deep_plot"), width=6)
            #)
        )
    ),
    
    ui.nav_panel(
      "Per-Class Insights",
      ui.div(
        [
    
          # === ROW 1: Model Selector ===
          ui.card(
              ui.h4("Model Variant", style="font-size: 1.1rem; font-weight: 600;"),
              ui.input_radio_buttons(
                  "selected_model",
                  "",
                  choices=["RF (PCA)", "XGBoost (PCA)", "CNN", "ResNet"],
                  selected="RF (PCA)",
                  inline=True
              ),
              style="margin-bottom: 1rem;"
          ),
    
          # === ROW 2: Cell Group + Metric + Heatmap + Barplot ===
          ui.div([
              ui.card(
                  ui.div([
                      ui.h4("Cell Group", style="font-size: 1.1rem; font-weight: 600;"),
                      ui.input_radio_buttons("selected_class", "", choices=["Tumour", "Immune", "Stromal", "Other"], selected="Tumour", inline=False),
                      ui.h4("Metric", style="font-size: 1.1rem; font-weight: 600;"),
                      ui.input_radio_buttons("heatmap_metric", "", choices=["Precision", "Recall", "Confidence"], selected="Precision", inline=False),
                  ]),
                  style="flex: 0.75;"
              ),
              ui.card(ui.output_plot("heatmap"), style="flex: 2.125;"),
              ui.card(ui.output_plot("per_class_metric_plot"), style="flex: 2.125;"),
          ], style="display: flex; gap: 1rem; margin-bottom: 1rem;"),
    
          # === ROW 3: Confusion Matrix + Upload Prediction ===
          ui.div([
              ui.card(
                  ui.div([
                      ui.h4("Confusion Matrix Settings", style="font-size: 1.1rem; font-weight: 600;"),
                      ui.input_radio_buttons("cm_blur", "Blur Level", choices=[0, 1, 3, 5, 7, 9, 19], selected=0, inline=False),
                      ui.input_radio_buttons("cm_noise", "Noise Level", choices=[0, 1, 3, 5, 10, 20, 30], selected=0, inline=False),
                  ]),
                  style="flex: 0.75;"
              ),
              ui.card(ui.output_plot("confusion_matrix_plot"), style="flex: 2.125;"),
              ui.card(
                  ui.div([
                      ui.h4("Upload Your Own Image", style="font-size: 1.1rem; font-weight: 600;"),
                      ui.input_file("user_image", "Choose a PNG Image", accept=[".png"]),
                      ui.output_plot("user_prediction_heatmap")
                  ]),
                  style="flex: 2.125;"
              ),
          ], style="display: flex; gap: 1rem;")
        ]
      )
    ),

    title="Image Model Dashboard"
)

def server(input, output, session):

    # Reusable helper function
    def plot_metric_bar(metric_name: str, title: str, ylabel: str):
        subset = get_filtered_df()
    
        if subset.empty:
            plt.figure()
            plt.text(0.5, 0.5, "No data available", ha='center', va='center')
            plt.axis("off")
            return plt.gcf()
    
        subset = subset.sort_values(metric_name, ascending=False)
        colors = [model_colors.get(label, "#cccccc") for label in subset["Model_Label"]]
    
        plt.figure(figsize=(6, 5))
        bars = plt.bar(subset["Model_Label"], subset[metric_name], color=colors)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
        plt.title(title)
    
        for bar, val in zip(bars, subset[metric_name]):
            plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2f}", ha='center')
    
        plt.ylim(0, min(1.0, max(subset[metric_name].max() + 0.05, 0.1)))
        plt.tight_layout()
        return plt.gcf()
  
    def get_filtered_df():
        """Returns the part of df matching the current blur and noise selections."""
        blur = int(input.blur())
        noise = float(input.noise())
        return df[(df["blur_size"] == blur) & (df["noise_level"] == noise)]

    def make_mock_plot(title):
        x = np.arange(4)
        y = np.random.rand(4)
        plt.figure(figsize=(4, 2))
        plt.bar(x, y, color="skyblue")
        plt.title(title)
        return plt.gcf()

    @output
    @render.plot
    def accuracy_plot():
        subset = get_filtered_df()
    
        if subset.empty:
            plt.figure()
            plt.text(0.5, 0.5, "No data for selected settings", ha='center', va='center')
            plt.axis("off")
            return plt.gcf()
    
        # Get colors from model_colors
        colors = [model_colors.get(label, "#cccccc") for label in subset["Model_Label"]]
    
        plt.figure(figsize=(6, 5))
        bars = plt.bar(subset["Model_Label"], subset["accuracy"], color=colors)
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45, ha="right")
        plt.title("Accuracy by Model")
    
        # Show bar values
        for bar, val in zip(bars, subset["accuracy"]):
            plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2f}", ha='center')
    
        # Prevent clipping
        plt.ylim(0, min(1.0, max(subset["accuracy"].max() + 0.05, 0.1)))
        plt.tight_layout()
        return plt.gcf()

    @output
    @render.plot
    def precision_plot():
        subset = get_filtered_df()
    
        if subset.empty:
            plt.figure()
            plt.text(0.5, 0.5, "No data for selected settings", ha='center', va='center')
            plt.axis("off")
            return plt.gcf()
    
        subset = subset.sort_values("precision", ascending=False)
        colors = [model_colors.get(label, "#cccccc") for label in subset["Model_Label"]]
    
        plt.figure(figsize=(6, 5))
        bars = plt.bar(subset["Model_Label"], subset["precision"], color=colors)
        plt.ylabel("Precision")
        plt.xticks(rotation=45, ha="right")
        plt.title("Precision by Model")
    
        for bar, val in zip(bars, subset["precision"]):
            plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2f}", ha='center')
    
        plt.ylim(0, min(1.0, max(subset["precision"].max() + 0.05, 0.1)))
        plt.tight_layout()
        return plt.gcf()

    @output
    @render.plot
    def f1_plot():
        subset = get_filtered_df()
    
        if subset.empty:
            plt.figure()
            plt.text(0.5, 0.5, "No data for selected settings", ha='center', va='center')
            plt.axis("off")
            return plt.gcf()
    
        subset = subset.sort_values("f1", ascending=False)
        colors = [model_colors.get(label, "#cccccc") for label in subset["Model_Label"]]
    
        plt.figure(figsize=(6, 5))
        bars = plt.bar(subset["Model_Label"], subset["f1"], color=colors)
        plt.ylabel("F1 Score")
        plt.xticks(rotation=45, ha="right")
        plt.title("F1 Score by Model")
    
        for bar, val in zip(bars, subset["f1"]):
            plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2f}", ha='center')
    
        plt.ylim(0, min(1.0, max(subset["f1"].max() + 0.05, 0.1)))
        plt.tight_layout()
        return plt.gcf()

    @output
    @render.plot
    def confidence_plot():
        subset = get_filtered_df()
    
        if subset.empty:
            plt.figure()
            plt.text(0.5, 0.5, "No data for selected settings", ha='center', va='center')
            plt.axis("off")
            return plt.gcf()
    
        subset = subset.sort_values("avg_conf", ascending=False)
        colors = [model_colors.get(label, "#cccccc") for label in subset["Model_Label"]]
    
        plt.figure(figsize=(6, 5))
        bars = plt.bar(subset["Model_Label"], subset["avg_conf"], color=colors)
        plt.ylabel("Average Confidence")
        plt.xticks(rotation=45, ha="right")
        plt.title("Confidence by Model")
    
        for bar, val in zip(bars, subset["avg_conf"]):
            plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.0%}", ha='center')
    
        plt.ylim(0, min(1.0, max(subset["avg_conf"].max() + 0.05, 0.1)))
        plt.tight_layout()
        return plt.gcf()
          
    @output
    @render.plot
    def selected_metric_plot():
        metric = input.line_metric()
        if metric == "F1 Score":
            return plot_metric_bar("f1", "F1 Score by Model", "F1 Score")
        elif metric == "Precision":
            return plot_metric_bar("precision", "Precision by Model", "Precision")
        elif metric == "Recall":
            return plot_metric_bar("recall", "Recall by Model", "Recall")
        elif metric == "Accuracy":
            return plot_metric_bar("accuracy", "Accuracy by Model", "Accuracy")

    @output
    @render.plot
    def precision_per_class_plot():
        blur = int(input.blur())
        noise = float(input.noise())
        model = input.selected_model()
    
        subset = df[
            (df["blur_size"] == blur) &
            (df["noise_level"] == noise) &
            (df["Model_Label"] == model)
        ]
    
        if subset.empty:
            plt.figure()
            plt.text(0.5, 0.5, "No data available", ha='center', va='center')
            plt.axis("off")
            return plt.gcf()
    
        class_order = ["Tumour", "Immune", "Stromal", "Other"]
        values = [subset[f"precision_{cls}"].values[0] for cls in class_order]
        colors = [class_colors[cls] for cls in class_order]
    
        plt.figure(figsize=(6, 4))
        bars = plt.bar(class_order, values, color=colors)
        plt.ylim(0, max(values) + 0.1)
        plt.ylabel("Precision Per Class")
        #plt.title(f"Precision — {model}")
    
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
        plt.tight_layout()
        return plt.gcf()
      
    @output
    @render.plot
    def recall_per_class_plot():
        blur = int(input.blur())
        noise = float(input.noise())
        model = input.selected_model()
    
        subset = df[
            (df["blur_size"] == blur) &
            (df["noise_level"] == noise) &
            (df["Model_Label"] == model)
        ]
    
        if subset.empty:
            plt.figure()
            plt.text(0.5, 0.5, "No data available", ha='center', va='center')
            plt.axis("off")
            return plt.gcf()
    
        class_order = ["Tumour", "Immune", "Stromal", "Other"]
        values = [subset[f"recall_{cls}"].values[0] for cls in class_order] 
        colors = [class_colors[cls] for cls in class_order]
    
        plt.figure(figsize=(6, 4))
        bars = plt.bar(class_order, values, color=colors)
        plt.ylim(0, max(values) + 0.1)
        plt.ylabel("Recall Per Class")
        #plt.title(f"F1 — {model}")
    
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
        plt.tight_layout()
        return plt.gcf()

    @output
    @render.plot
    def f1_per_class_plot():
        blur = int(input.blur())
        noise = float(input.noise())
        model = input.selected_model()
    
        subset = df[
            (df["blur_size"] == blur) &
            (df["noise_level"] == noise) &
            (df["Model_Label"] == model)
        ]
    
        if subset.empty:
            plt.figure()
            plt.text(0.5, 0.5, "No data available", ha='center', va='center')
            plt.axis("off")
            return plt.gcf()
    
        class_order = ["Tumour", "Immune", "Stromal", "Other"] 
        values = [subset[f"f1_{cls}"].values[0] for cls in class_order] 
        colors = [class_colors[cls] for cls in class_order]
    
        plt.figure(figsize=(6, 4))
        bars = plt.bar(class_order, values, color=colors)
        plt.ylim(0, max(values) + 0.1)
        plt.ylabel("F1 Per Class")
        #plt.title(f"F1 — {model}")
    
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
        plt.tight_layout()
        return plt.gcf()

    @output
    @render.plot
    def heatmap():
        model = input.selected_model()
        class_name = input.selected_class().lower()  # FIXED HERE
        metric = input.heatmap_metric().lower()  # "precision", "recall", or "confidence"
    
        if metric == "confidence":
            col = f"confidence_{class_name}_avg"  # FIXED to match column naming
        else:
            col = f"{metric}_{class_name}"
    
        subset = df[df["Model_Label"] == model]
        if subset.empty or col not in subset.columns:
            plt.figure()
            plt.text(0.5, 0.5, "No data available", ha='center', va='center')
            plt.axis("off")
            return plt.gcf()
    
        pivot = subset.pivot(index="blur_size", columns="noise_level", values=col)
    
        plt.figure(figsize=(8, 5))
        plt.imshow(pivot, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto", origin="lower")
    
        plt.title(f"{metric.capitalize()} (%) for {class_name.capitalize()} — {model}")
        plt.xlabel("Noise")
        plt.ylabel("Blur")
        plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns)
        plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
    
        for i, blur_val in enumerate(pivot.index):
            for j, noise_val in enumerate(pivot.columns):
                value = pivot.loc[blur_val, noise_val]
                if not pd.isna(value):
                    plt.text(j, i, f"{value * 100:.0f}%", ha="center", va="center", color="black")
    
        plt.tight_layout()
        return plt.gcf()
        
    @output
    @render.plot
    def line_metric_plot():
        metric_lookup = {
            "Accuracy": "accuracy",
            "Precision": "precision",
            "Recall": "recall",
            "F1 Score": "f1"
        }
        metric_col = metric_lookup[input.line_metric()]
        
        vary = input.fixed_aug()  # either "Blur" or "Noise"
        vary_col = "blur_size" if vary == "Blur" else "noise_level"
        fixed_col = "noise_level" if vary == "Blur" else "blur_size"
        fixed_val = int(input.noise()) if fixed_col == "noise_level" else int(input.blur())
    
        plt.figure(figsize=(8, 5))
    
        for model in df["Model_Label"].unique():
            subset = df[(df["Model_Label"] == model) & (df[fixed_col] == fixed_val)]
            if subset.empty:
                continue
    
            subset = subset.sort_values(by=vary_col)
            plt.plot(
                subset[vary_col],
                subset[metric_col],
                marker="o",
                label=model,
                color=model_colors.get(model, "#999999")
            )
    
        xticks = noise_levels if vary_col == "noise_level" else blur_levels
        plt.xticks(xticks)
        
        plt.title(f"{input.line_metric()} vs {vary} at {fixed_col.replace('_', ' ').title()} = {fixed_val}")
        plt.xlabel(vary)
        plt.ylabel(input.line_metric())
        plt.ylim(0, 1.0)
        plt.grid(True)
        plt.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=len(df["Model_Label"].unique()),
            frameon=False
        )
        plt.tight_layout()
        return plt.gcf()
        
    @output
    @render.plot
    def boxplot():
        model = input.selected_model()
        class_name = input.selected_class()
        facet_by = input.boxplot_metric().capitalize()  # "Blur" or "Noise"
        fixed_by = "Noise" if facet_by == "Blur" else "Blur"
    
        mean_col = f"avg_conf_{class_name}"
        std_col = f"std_conf_{class_name}"
    
        subset = df[df["Model_Label"] == model].copy()
        if subset.empty or mean_col not in subset.columns:
            plt.figure()
            plt.text(0.5, 0.5, "No data available", ha='center', va='center')
            plt.axis("off")
            return plt.gcf()
    
        facet_values = sorted(subset[facet_by].unique())
        fixed_values = sorted(subset[fixed_by].unique())
    
        n_facets = len(facet_values)
        ncols = 4
        nrows = int(np.ceil(n_facets / ncols))
    
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
        axs = axs.flatten()  # flatten for easier indexing
    
        for i, val in enumerate(facet_values):
            ax = axs[i]
            data = subset[subset[facet_by] == val]
            x_vals = fixed_values
            x_labels = [str(v) for v in x_vals]
            means = []
            stds = []
    
            for fval in x_vals:
                row = data[data[fixed_by] == fval]
                if not row.empty:
                    means.append(row[mean_col].values[0])
                    stds.append(row[std_col].values[0])
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
    
            means = np.array(means)
            stds = np.array(stds)
            x_ticks = np.arange(len(x_vals))
    
            ax.plot(x_ticks, means, marker='o', color=class_colors[class_name], label="Mean")
            ax.fill_between(x_ticks, means - stds, means + stds, color=class_colors[class_name], alpha=0.3, label="± SD")
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax.set_title(f"{facet_by} = {val}")
            ax.set_xlabel(fixed_by)
            ax.set_ylim(0, 1)
    
        # Hide unused subplots
        for ax in axs[n_facets:]:
            ax.set_visible(False)
    
        axs[0].set_ylabel("Confidence")
        fig.suptitle(f"{model} — Avg {class_name} Confidence ± SD", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig
      
    @output
    @render.plot
    def per_class_metric_plot():
        blur = int(input.blur())
        noise = float(input.noise())
        model = input.selected_model()
        class_name = input.selected_class()
        metric = input.heatmap_metric().lower()
    
        class_order = ["tumour", "immune", "stromal", "other"]
        display_order = ["Tumour", "Immune", "Stromal", "Other"]
    
        subset = df[
            (df["blur_size"] == blur) &
            (df["noise_level"] == noise) &
            (df["Model_Label"] == model)
        ]
    
        if subset.empty:
            plt.figure()
            plt.text(0.5, 0.5, "No data available", ha='center', va='center')
            plt.axis("off")
            return plt.gcf()
    
        if metric == "confidence":
            means = [subset[f"confidence_{cls}_avg"].values[0] for cls in class_order]
            stds  = [subset[f"confidence_{cls}_std"].values[0] for cls in class_order]
            ylabel = "Confidence ± SD"
        else:
            means = [subset[f"{metric}_{cls}"].values[0] for cls in class_order]
            stds = None
            ylabel = f"{metric.capitalize()} Per Class"
    
        colors = [class_colors[cls.capitalize()] for cls in class_order]
    
        plt.figure(figsize=(6, 4))
        bars = plt.bar(display_order, means, yerr=stds if stds else None, color=colors, capsize=5)
        plt.ylim(0, max(np.array(means) + (np.array(stds) if stds else 0)) + 0.1)
        plt.ylabel(ylabel)
        plt.title(f"{model} — {metric.capitalize()} by Class")
    
        for i, v in enumerate(means):
            offset = stds[i] if stds else 0
            plt.text(i, v + offset + 0.01, f"{v:.2f}", ha='center')
    
        plt.tight_layout()
        return plt.gcf()
      
    @output
    @render.plot
    def confusion_matrix_plot():
        import ast
        from sklearn.metrics import ConfusionMatrixDisplay
    
        model = input.selected_model().strip().lower()
        blur = int(input.cm_blur())
        noise = float(input.cm_noise())
    
       # row = df[
            #(df["Model_Label"] == model) &
            #(df["blur_size"] == blur) &
            #(df["noise_level"] == noise)
        #]
        row = df[
            (df["Model_Label"].str.strip().str.lower() == model) &
            (df["blur_size"] == blur) &
            (df["noise_level"] == noise)
        ]
    
        if row.empty or "confusion_matrix" not in row.columns:
            plt.figure()
            plt.text(0.5, 0.5, "No data for selected blur/noise", ha='center', va='center')
            plt.axis("off")
            return plt.gcf()
    
        try:
            cm = np.array(ast.literal_eval(row["confusion_matrix"].values[0]))
        except Exception as e:
            plt.figure()
            plt.text(0.5, 0.5, f"Error parsing matrix: {e}", ha='center', va='center')
            plt.axis("off")
            return plt.gcf()
    
        labels = ["Tumour", "Immune", "Stromal", "Other"]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        plt.title(f"Confusion Matrix — {model} (Blur {blur}, Noise {noise})")
        return fig
      
    @output
    @render.plot
    def user_prediction_heatmap():
        file = input.user_image()
        if file is None:
            plt.figure()
            plt.text(0.5, 0.5, "Upload an image to view predictions", ha='center', va='center')
            plt.axis("off")
            return plt.gcf()
    
        img_path = file[0]['datapath']
        img = shiny_data.load_resize(img_path)
        model_name = input.selected_model()
    
        if model_name == "CNN":
            preds, blur_sizes, noise_levels = predict_cnn(img, cnn_model)
        elif model_name == "XGBoost (PCA)":
            preds, blur_sizes, noise_levels = predict_xgb(img, pca, xgb_model)
        elif model_name == "ResNet":
            preds, blur_sizes, noise_levels = predict_resnet(img, rn_model)
        elif model_name == "RF (PCA)":
            preds, blur_sizes, noise_levels = predict_rf(img, pca, rf_model)
        else:
            plt.figure()
            plt.text(0.5, 0.5, "Model not supported for prediction", ha='center', va='center')
            plt.axis("off")
            return plt.gcf()
    
        class_names = ['Immune', 'Other', 'Stromal', 'Tumour']
        class_colors = {
            0: "#2ca02c",  # Immune
            1: "#ff7f0e",  # Other
            2: "#1f77b4",  # Stromal
            3: "#d62728",  # Tumour
        }
    
        plt.figure(figsize=(10, 6))
        color_matrix = np.vectorize(class_colors.get)(preds)
        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, color=class_colors[preds[i, j]]))
                plt.text(j + 0.5, i + 0.5, class_names[preds[i, j]][0], ha='center', va='center', color='white', fontsize=10)
    
        plt.xticks(np.arange(len(blur_sizes)) + 0.5, blur_sizes)
        plt.yticks(np.arange(len(noise_levels)) + 0.5, noise_levels)
        plt.gca().invert_yaxis()
        plt.gca().set_xticks(np.arange(len(blur_sizes)), minor=True)
        plt.gca().set_yticks(np.arange(len(noise_levels)), minor=True)
        plt.grid(True, which='minor', color='gray', linewidth=0.5)
        plt.xlabel("Blur Size")
        plt.ylabel("Noise Level")
        plt.title(f"Prediction Heatmap ({model_name})")
    
        # Add manual legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=class_colors[i]) for i in range(4)]
        # plt.legend(handles, class_names, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
    
        # Set ticks centered in each cell
        plt.xticks(np.arange(len(blur_sizes)) + 0.5, blur_sizes)
        plt.yticks(np.arange(len(noise_levels)) + 0.5, noise_levels)
        
        # Expand the axes limits so the last column isn't cut off
        plt.xlim(0, len(blur_sizes))
        plt.ylim(0, len(noise_levels))
        
        # Flip y-axis for correct orientation
        plt.gca().invert_yaxis()

        plt.tight_layout()
        return plt.gcf()

    # === Image Outputs ===
    @output
    @render.image
    def tumour_img():
        blur = int(input.blur())
        noise = int(input.noise())
        filename = input.tumour_cell()
        path = f"example_images_augmented/combo/{blur}_blur_{noise}_noise/{filename}"
        return {
            "src": path,
            "alt": "Tumour Cell Image",
            "width": "100%",
            "height": "60%"
        }

    @output
    @render.image
    def immune_img():
        blur = int(input.blur())
        noise = int(input.noise())
        filename = input.immune_cell()
        path = f"example_images_augmented/combo/{blur}_blur_{noise}_noise/{filename}"
        return {
            "src": path,
            "alt": "Immune Cell Image",
            "width": "100%",
            "height": "60%"
        }
        

    @output
    @render.image
    def stromal_img():
        blur = int(input.blur())
        noise = int(input.noise())
        filename = input.stromal_cell()
        path = f"example_images_augmented/combo/{blur}_blur_{noise}_noise/{filename}"
        return {
            "src": path,
            "alt": "Stromal Cell Image",
            "width": "100%",
            "height": "60%"
        }
        
    @output
    @render.image
    def nonstromal_img():
        blur = int(input.blur())
        noise = int(input.noise())
        filename = input.nonstromal_cell()
        path = f"example_images_augmented/combo/{blur}_blur_{noise}_noise/{filename}"
        return {
            "src": path,
            "alt": "Non-Stromal Cell Image",
            "width": "100%",
            "height": "60%"
        }
        
from pathlib import Path
app = App(app_ui, server)
