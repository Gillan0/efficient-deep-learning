import pandas as pd
import matplotlib.pyplot as plt

# Load CSV (adjust path if needed)
NAME = "cosine-mixup_pruned2"

DIR = ""

df = pd.read_csv(f"./src/lab4/logs/{NAME}.log")

heatmap_data = df.pivot(
    index="linear_perc",
    columns="conv_perc",
    values="pruned_acc"
)

# Sort axes (important for proper visualization)
heatmap_data = heatmap_data.sort_index()
heatmap_data = heatmap_data.sort_index(axis=1)

# === 3. Convert to numpy array ===
Z = heatmap_data.values
x = heatmap_data.columns.values
y = heatmap_data.index.values

# === 4. Plot heatmap ===
plt.figure()

plt.imshow(
    Z,
    origin="lower",
    aspect="auto",
    extent=[x.min(), x.max(), y.min(), y.max()]
)

plt.colorbar(label="Test Accuracy (%)")

plt.xlabel("Convolutional Percentage")
plt.ylabel("Linear Percentage")
plt.title("Accuracy Heatmap")

# Save figure
plt.savefig(f"./src/lab4/logs/{NAME}.png", dpi=300)

# Close figure (good practice for scripts)
plt.close()
