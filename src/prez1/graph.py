import pandas as pd
import matplotlib.pyplot as plt

# Load CSV (adjust path if needed)
NAME = "lr_small"

df = pd.read_csv(f"src/prez1/resnet_test/{NAME}.log")

# Create plot
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_loss"], label="Training Loss", marker="o")
plt.plot(df["epoch"], df["test_loss"], label="Testing Loss", marker="o")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Testing Loss over Epochs")
plt.legend()
plt.grid(True)

plt.xticks(df["epoch"])

plt.tight_layout()

# Save figure
plt.savefig(f"src/prez1/resnet_test/{NAME}.png", dpi=300)

# Close figure (good practice for scripts)
plt.close()
