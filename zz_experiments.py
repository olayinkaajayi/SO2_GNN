import matplotlib.pyplot as plt
import json
import numpy as np
import os.path as osp

folder = "zz_ablation_CV_accuracy"
file = "aug-angles_cv_accuracy.json"
# Path to your JSON file
result_path = osp.join(folder, file)

# Load JSON
with open(result_path, "r") as f:
    data = json.load(f)

# Data
proportion = np.array(data["angle"])
proportion = np.rad2deg(proportion)
# extra = np.array([[0.79], 
#                    [84.22]
#                    ])
data_1 = np.array(data["GCNoT-AU"])
data_2 = np.array(data["InfoGCN-AU"])


# NeurIPS-style settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "lines.linewidth": 2,
    "lines.markersize": 7,
})

fig, ax = plt.subplots(figsize=(6.5, 4.5))

# Plot
ax.plot(proportion, data_1, marker='o', linestyle='-', label=r"GCN$o\mathcal{T}$ A/U")
ax.plot(proportion, data_2, marker='^', linestyle='--', label='InfoGCN A/U')
# ax.scatter(np.rad2deg(extra[0]), extra[1], marker='*', label=r"GCN$o\mathcal{T}$ (9 par) A/U")

# Labels
ax.set_xlabel(r'$\pm$ angle range (degrees)')
ax.set_ylabel('Cross-View Accuracy (%)')

# Ticks and limits
# ax.set_xticks(proportion)
ax.set_ylim(10, 95)

# Grid
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# Remove top and right spines (key change)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Optional: slightly emphasize left/bottom spines
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)

# Legend
ax.legend(frameon=True)

plt.tight_layout()

# Save
plt.savefig(osp.join(folder,"angles-cross_view_accuracy-extra.png"), dpi=300, bbox_inches='tight')
