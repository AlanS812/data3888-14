import pandas as pd
import matplotlib.pyplot as plt

# Define the model names and corresponding CSV files
models = {
    "Resnet-50": "resnet50_augmented_metrics.csv",
    "XGBoost": "xgboost_augmented_metrics.csv",
    #"Random Forest": "model_results_RF.csv",
    #"CNN": "model4.csv"
}

# Set the noise level you're interested in
target_noise_level = 1

plt.figure(figsize=(10, 6))

# Loop through models and plot accuracy vs. blur_size
for model_name, file in models.items():
    df = pd.read_csv(file)
    
    # Filter by noise level
    filtered = df[df["noise"] == target_noise_level]
    
    # Sort for cleaner lines
    filtered = filtered.sort_values(by="blur")
    
    plt.plot(filtered["blur"], filtered["accuracy"], marker='o', label=model_name)

    print("Completed: ", model_name)

plt.title(f"Accuracy vs. Blur Size at Noise Level {target_noise_level}")
plt.xlabel("Blur Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()