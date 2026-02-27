import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
NAME = "cosine-mixup_pruned_retrained_archive"
df = pd.read_csv(f"./src/lab4/logs/{NAME}.log")

plt.figure(figsize=(10, 6))

plt.plot(
    df["conv_perc"],
    df["pruned_acc"],
    marker="o",
    markersize=6,
    linewidth=2,
    label="Test Accuracy After Pruning"
)

plt.plot(
    df["conv_perc"],
    df["test_acc"],
    marker="s",
    markersize=6,
    linewidth=2,
    label="Test Accuracy After Pruning and 10 Epochs of Fine-Tuning"
)

plt.axhline(
    y=90,
    color="red",
    linestyle="--",
    linewidth=1.8,
    label="Target Accuracy (90%)"
)

plt.xlabel("Proportion of Convolutional Layers removed (%)", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title(
    "Impact of Convolutional Layer Pruning on Model Accuracy\n"
    "(Linear Layer Proportion = 0.6)",
    fontsize=14
)

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(frameon=True)
plt.tight_layout()

plt.savefig(f"./src/lab4/logs/{NAME}.png", dpi=300)
plt.close()