import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Model names
models = ["adam", "cosine", "sgd-plateau"]

# Base colors for train and test
train_base_color = np.array([0.2, 0.4, 0.8])  # blue
test_base_color = np.array([1.0, 0.5, 0.0])   # orange

# Shades per model (light to dark)
shades = np.linspace(0.5, 1.0, len(models))

plt.figure(figsize=(10, 6))

for i, model in enumerate(models):
    df = pd.read_csv(f"src/prez1/resnet_test/{model}.log")
    
    # Adjust color shade per model
    train_color = train_base_color * shades[i]
    test_color = test_base_color * shades[i]
    
    # Plot training loss
    plt.plot(df["epoch"], df["train_loss"], label=f"{model} Train", color=train_color, marker="o", linestyle='-')
    
    # Plot testing loss
    plt.plot(df["epoch"], df["test_loss"], label=f"{model} Test", color=test_color, marker="s", linestyle='--')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Testing Loss for Different Optimizers (Shaded by Model)")
plt.legend()
plt.grid(True)
plt.xticks(df["epoch"])
plt.tight_layout()

# Save figure
plt.savefig("src/prez1/resnet_test/all_models_comparison.png", dpi=300)
plt.close()
