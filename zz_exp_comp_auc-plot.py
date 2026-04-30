import matplotlib.pyplot as plt
import json
import numpy as np
import os.path as osp

# =======================
# Load data
# =======================
folder = "zz_ablation_CV_accuracy"
file = "aug-angles_cv_accuracy.json"
result_path = osp.join(folder, file)

with open(result_path, "r") as f:
    data = json.load(f)

# Data
x = np.array(data["angle"])
x = np.rad2deg(x)

y1 = np.array(data["GCNoT-AU"])
y2 = np.array(data["InfoGCN-AU"])
extra = np.array([[0.79, 1.57], 
                   [84.22, 87.84]
                   ])

# =======================
# Interpolation (for smooth shading)
# =======================
x_dense = np.linspace(x.min(), x.max(), 300)

y1_dense = np.interp(x_dense, x, y1)
y2_dense = np.interp(x_dense, x, y2)

# =======================
# Normalize to [0,1]
# =======================
def minmax_norm(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

x_norm = minmax_norm(x)
y1_norm = minmax_norm(y1)
y2_norm = minmax_norm(y2)

# =======================
# AUC (original spacing)
# =======================
auc1 = np.trapezoid(y1_norm, x_norm)
auc2 = np.trapezoid(y2_norm, x_norm)


# =======================
# Plot
# =======================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "legend.fontsize": 13,
})

fig, ax = plt.subplots(figsize=(6.5, 4.5))

# Lines
line1, = ax.plot(x, y1, marker='o', linestyle='-', label=r"GCN$o\mathcal{T}$ A/U")
line2, = ax.plot(x, y2, marker='^', linestyle='--', label='InfoGCN A/U')
ax.scatter(np.rad2deg(extra[0]), extra[1], marker='*', label=r"GCN$o\mathcal{T}$ (9 par) A/U")

# Shaded areas
ax.fill_between(x_dense, y1_dense,
                color=line1.get_color(),
                alpha=0.15)

ax.fill_between(x_dense, y2_dense,
                color=line2.get_color(),
                alpha=0.15)

# Labels
ax.set_xlabel(r'$\pm$ angle range (degrees)')
ax.set_ylabel('Cross-View Accuracy (%)')
ax.set_ylim(10, 95)

# Grid & style
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# =======================
# Annotate AUC values
# =======================
# Choose reasonable positions (mid-region of each curve)
x_text1 = x_dense[len(x_dense)//3]
y_text1 = np.interp(x_text1, x_dense, y1_dense)

x_text2 = x_dense[2*len(x_dense)//3]
y_text2 = np.interp(x_text2, x_dense, y2_dense)

ax.text(x_text1, y_text1,
        f"AUC = {auc1:.2f}",
        fontsize=12)

ax.text(x_text2, y_text2,
        f"AUC = {auc2:.2f}",
        fontsize=12)

# Legend
ax.legend(frameon=True)

plt.tight_layout()

# Save
plt.savefig(osp.join(folder, "angles_with_shaded_auc-extra.png"),
            dpi=300, bbox_inches='tight')
