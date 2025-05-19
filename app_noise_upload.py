from shiny import App, ui, render
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

df = pd.read_csv("results.csv")

# defining augmentations
blur_levels = [0, 1, 3, 5, 7, 10]
contrast_levels = [0, 1.25, 1.5, 1.75, 2.0]
model_colors = {
    "RF (pixels)": "#1f77b4",
    "RF (PCA)": "#ff7f0e",
    "XGB (pixels)": "#2ca02c",
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
          # === AUGMENTATION + IMAGE CARDS SIDE-BY-SIDE ===
          ui.div(
              # Left: Augmentation panel
              ui.div(
                  ui.card(
                      ui.h4("Augmentation Settings"),
                      ui.input_radio_buttons("blur", "Select Blur Level", choices=blur_levels, inline=True),
                      ui.input_radio_buttons("contrast", "Select Contrast Level", choices=contrast_levels, inline=True),
                  ),
                  style="width: 25%; padding-right: 1rem;"
              ),
  
              # Right: Image cards
              ui.div(
                  *[
                      ui.card(
                          ui.div(
                            ui.h5(title),
                            ui.div(
                                ui.output_image(image_id),
                                style="width: 100%; max-height: 250px; overflow: hidden;"
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
                          ("Tumour Cell", "tumour_img", "tumour_cell", tumour_labels),
                          ("Immune Cell", "immune_img", "immune_cell", immune_labels),
                          ("Stromal Cell", "stromal_img", "stromal_cell", stromal_like_labels),
                          ("Other Cell", "nonstromal_img", "nonstromal_cell", non_stromal_other_labels),
                      ]
                  ],
                  style="width: 75%; display: flex; flex-wrap: nowrap; gap: 1rem; justify-content: space-between;"
              ),
              style="display: flex;"
          ),
  
          # === METRIC PLOTS BELOW ===
          ui.layout_columns(
              ui.card(ui.h3("Accuracy"), ui.output_plot("accuracy_plot"), width=4),
              ui.card(ui.h4("Precision"), ui.output_plot("precision_plot"), width=4),
              ui.card(ui.h4("F1 Score"), ui.output_plot("f1_plot"), width=4)
          ),
          ui.layout_columns(
               ui.card(ui.h4("Average Confidence"), ui.output_plot("confidence_plot")),
               ui.card(ui.h4("Training and Validation"), ui.output_plot("deep_plot"))
          ),
      )
  ),
    
    ui.nav_panel(
      "Per-Class Insights",
        ui.div(
            # Custom 30/70 layout for top row
            ui.div(
                ui.div(
                    ui.card(
                        ui.h4("Select a Model Variant"),
                        ui.input_radio_buttons(
                            "selected_model",
                            "",
                            choices=["RF (pixels)", "RF (PCA)", "XGB (pixels)", "XGB (PCA)", "CNN", "ResNet"],
                            selected="RF (pixels)"
                        ),
                        ui.input_file(
                          "uploaded_image",
                          "Upload a PNG Image for Testing",
                          accept=[".png"]
                      )
                    ),
                    style="width: 30%; padding-right: 1rem;"
                ),
                ui.div(
                    ui.card(
                        ui.h4("Most Confident Class Heatmap"),
                        ui.output_plot("heatmap")
                    ),
                    style="width: 70%;"
                ),
                style="display: flex;"
            ),
    
            # Bottom row stays the same
            ui.layout_columns(
                ui.card(
                    ui.h4("Precision per class"),
                    ui.output_plot("precision_per_class_plot"),
                    width=4
                ),
                ui.card(
                    ui.h4("F1 per class"),
                    ui.output_plot("f1_per_class_plot"),
                    width=4
                ),
                ui.card(
                    ui.h4("Confidence per class"),
                    ui.output_plot("conf_per_class_plot"),
                    width=4
                ),
                style="margin-top: 1rem;"
            )
        )
  ),

    title="Image Model Dashboard"
    #style="font-size: 0.9rem;"  # ✅ Move this here
)

def server(input, output, session):
    def get_filtered_df():
        """Returns the part of df matching the current blur and contrast selections."""
        blur = int(input.blur())
        contrast = float(input.contrast())
        return df[(df["Blur"] == blur) & (df["Contrast"] == contrast)]

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
    def precision_per_class_plot():
        blur = int(input.blur())
        contrast = float(input.contrast())
        model = input.selected_model()
    
        subset = df[
            (df["Blur"] == blur) &
            (df["Contrast"] == contrast) &
            (df["Model_Label"] == model)
        ]
    
        if subset.empty:
            plt.figure()
            plt.text(0.5, 0.5, "No data available", ha='center', va='center')
            plt.axis("off")
            return plt.gcf()
    
        class_order = ["Immune", "Other", "Stromal", "Tumour"]
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
    def f1_per_class_plot():
        blur = int(input.blur())
        contrast = float(input.contrast())
        model = input.selected_model()
    
        subset = df[
            (df["Blur"] == blur) &
            (df["Contrast"] == contrast) &
            (df["Model_Label"] == model)
        ]
    
        if subset.empty:
            plt.figure()
            plt.text(0.5, 0.5, "No data available", ha='center', va='center')
            plt.axis("off")
            return plt.gcf()
    
        class_order = ["Immune", "Other", "Stromal", "Tumour"]
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
    def conf_per_class_plot():
        blur = int(input.blur())
        contrast = float(input.contrast())
        model = input.selected_model()
    
        subset = df[
            (df["Blur"] == blur) &
            (df["Contrast"] == contrast) &
            (df["Model_Label"] == model)
        ]
    
        if subset.empty:
            plt.figure()
            plt.text(0.5, 0.5, "No data available", ha='center', va='center')
            plt.axis("off")
            return plt.gcf()
    
        class_order = ["Immune", "Other", "Stromal", "Tumour"]
        values = [subset[f"avg_conf_{cls}"].values[0] for cls in class_order]
        colors = [class_colors[cls] for cls in class_order]
    
        plt.figure(figsize=(6, 4))
        bars = plt.bar(class_order, values, color=colors)
        plt.ylim(0, max(values) + 0.1)
        plt.ylabel("Average Confidence Per Class")
        #plt.title(f"Confidence — {model}")
    
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
        plt.tight_layout()
        return plt.gcf()

    @output
    @render.plot
    def deep_plot():
        return make_mock_plot("Other DL Metric")

    @output
    @render.plot
    def heatmap():
        model = input.selected_model()
    
        # Filter data for selected model
        model_df = df[df["Model_Label"] == model]
    
        if model_df.empty:
            plt.figure()
            plt.text(0.5, 0.5, "No data for selected model", ha='center', va='center')
            plt.axis("off")
            return plt.gcf()
    
        # === Consistent colors and labels ===
        blur_vals = sorted(df["Blur"].unique())
        contrast_vals = sorted(df["Contrast"].unique())
    
        label_map = {"Tumour": "T", "Immune": "I", "Other": "O", "Stromal": "S"}
        color_map = {
            "T": class_colors["Tumour"],
            "I": class_colors["Immune"],
            "O": class_colors["Other"],
            "S": class_colors["Stromal"]
        }
    
        fig, ax = plt.subplots(figsize=(len(blur_vals), len(contrast_vals)))
    
        for i, contrast in enumerate(contrast_vals):
            for j, blur in enumerate(blur_vals):
                row = model_df[(model_df["Blur"] == blur) & (model_df["Contrast"] == contrast)]
                if row.empty:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, color='lightgrey'))
                    continue
    
                row = row.iloc[0]
                confs = {
                    "Tumour": row["avg_conf_Tumour"],
                    "Immune": row["avg_conf_Immune"],
                    "Other": row["avg_conf_Other"],
                    "Stromal": row["avg_conf_Stromal"]
                }
    
                top_class = max(confs, key=confs.get)
                char = label_map[top_class]
                color = color_map[char]
    
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
                ax.text(j + 0.5, i + 0.5, char, ha='center', va='center', fontsize=12, color='white')
    
        # === Axis styling ===
        ax.set_xticks(np.arange(len(blur_vals)) + 0.5)
        ax.set_xticklabels(blur_vals)
        ax.set_yticks(np.arange(len(contrast_vals)) + 0.5)
        ax.set_yticklabels(contrast_vals)
        ax.set_xlabel("Blur")
        ax.set_ylabel("Contrast")
        ax.set_title(f"Most Confident Class — {model}")
        ax.set_xlim(0, len(blur_vals))
        ax.set_ylim(0, len(contrast_vals))
        ax.invert_yaxis()
        ax.set_aspect("equal")
    
        return fig

    # === Image Outputs ===
    @output
    @render.image
    def tumour_img():
        return {
            "src": f"images/{input.tumour_cell()}",
            "alt": "Tumour Cell Image",
            "width": "100%",  # fill the card width
            "height": "60%"
        }

    @output
    @render.image
    def immune_img():
        return {
            "src": f"images/{input.immune_cell()}",
            "alt": "Immune Cell Image",
            "width": "100%",  # fill the card width
            "height": "60%"
        }

    @output
    @render.image
    def stromal_img():
        return {
            "src": f"images/{input.stromal_cell()}",
            "alt": "Stromal Cell Image",
            "width": "100%",  # fill the card width
            "height": "60%"
        }

    @output
    @render.image
    def nonstromal_img():
        return {
            "src": f"images/{input.nonstromal_cell()}",
            "alt": "Other Cell Image",
            "width": "100%",  # fill the card width
            "height": "60%"
        }
        
from pathlib import Path
app = App(app_ui, server)


