import matplotlib.pyplot as plt
from collections import defaultdict

# Data: (Model, Score, Accuracy)
data = [
    ("cosine-mixup-60-83", 2.3250, 91.520),
    ("cosine-mixup-60-83 (half)", 1.1625, 91.450),

    ("adam-mixup-60-95", 2.0859, 90.81),
    ("adam-mixup-60-95 (half)", 1.0429, 89.770),

    ("cosine-mixup", 3.9795, 91.830),
    ("adam-mixup", 3.9795, 94.310),

    ("cosine-mixup (half)", 1.9897, 91.830),
    ("adam-mixup (half)", 1.9897, 94.260),

    ("mobileNet-adam-mixup (half)", 0.3678, 92.120),
    ("mobileNet-cosine-mixup (half)", 0.3678, 90.620),

    ("customNet-adam-mixup (half)", 1.1934, 93.700),
    ("customNet-cosine-mixup (half)", 1.1934, 89.750),

    ("mobileNet-adam-mixup-pruned-60-70", 0.4531, 89.480),
    ("mobileNet-adam-mixup-pruned-60-70 (half)", 0.2266, 89.460),

    ("mobileNet-adam-mixup-pruned-60-65", 0.4732, 90.420),
    ("mobileNet-adam-mixup-pruned-60-65 (half)", 0.2366, 90.320),

    ("lightNet-adam", 0.0968, 91.240),
    ("lightNet-adam (half)", 0.0484, 91.240),
    ("lightNet-adam (8bits)", 0.0242, 90.730),
    ("lightNet-adam (6bits)", 0.0181, 91.420),

    ("lightNetDepth", 0.0283, 90.400),
    ("lightNetDepth (8 bits)", 0.0071, 90.060),
    ("lightNetDepth (6 bits)", 0.0053, 90.440),
]

# Define architecture grouping
def get_arch(name):
    if "mobileNet" in name:
        return "mobileNet"
    if "customNet" in name:
        return "customNet"
    if "lightNetDepth" in name:
        return "lightNetDepth"
    if "lightNet" in name:
        return "lightNet"
    return "other"

# Color + marker per architecture
styles = {
    "mobileNet": ("o", None),
    "customNet": ("s", None),
    "lightNet": ("^", None),
    "lightNetDepth": ("D", None),
    "other": ("x", None),
}

# Group data
groups = defaultdict(list)
for name, score, acc in data:
    arch = get_arch(name)
    groups[arch].append((score, acc, name))

# Plot
plt.figure(figsize=(10, 6))

for arch, points in groups.items():
    marker, _ = styles.get(arch, ("x", None))
    
    scores = [p[0] for p in points]
    accs = [p[1] for p in points]

    plt.scatter(scores, accs, label=arch, marker=marker)

    # Annotate
    for score, acc, name in points:
        plt.annotate(name, (score, acc), fontsize=7)

# Log scale on x-axis
plt.xscale("log")

plt.xlabel("Score (log scale)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Score (Grouped by Architecture)")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("accuracy_vs_score_log_grouped.png", dpi=300, bbox_inches="tight")
plt.close()