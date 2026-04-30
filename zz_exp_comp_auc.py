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

# Raw data
x = np.array(data["angle"])          # radians
x = np.rad2deg(x)                   # convert to degrees

y1 = np.array(data["GCNoT-AU"])
y2 = np.array(data["InfoGCN-AU"])

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
# AUC (uniform interpolation - optional but recommended)
# =======================
x_uniform = np.linspace(0, 1, 200)

y1_interp = np.interp(x_uniform, x_norm, y1_norm)
y2_interp = np.interp(x_uniform, x_norm, y2_norm)

auc1_uniform = np.trapezoid(y1_interp, x_uniform)
auc2_uniform = np.trapezoid(y2_interp, x_uniform)

# =======================
# Print results
# =======================
print("=== AUC (normalized axes) ===")
print(f"GCNoT  (raw spacing): {auc1:.4f}")
print(f"InfoGCN(raw spacing): {auc2:.4f}")
print()
print(f"GCNoT  (uniform grid): {auc1_uniform:.4f}")
print(f"InfoGCN(uniform grid): {auc2_uniform:.4f}")
