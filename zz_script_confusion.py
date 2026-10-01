import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def ntu60_label_to_name(label: int) -> str:
    """
    Returns the NTU RGB+D 60 action name for a given label (0-59).

    Args:
        label (int): Action label in [0, 59]

    Returns:
        str: Action class name
    """
    ntu60_classes = [
        "drink water", #0
        "eat meal/snack", #1
        "brushing teeth", #2
        "brushing hair", #3
        "drop", #4
        "pickup", #5
        "throw", #6
        "sitting down", #7
        "standing up (from sitting position)", #8
        "clapping", #9
        "reading", #10
        "writing", #11
        "tear up paper", #12
        "wear jacket", #13
        "take off jacket", #14
        "wear a shoe", #15
        "take off a shoe", #16
        "wear on glasses", #17
        "take off glasses", #18
        "put on a hat/cap", #19
        "take off a hat/cap", #20
        "cheer up", #21
        "hand waving", #22
        "kicking something", #23
        "reach into pocket", #24
        "hopping (one foot jumping)", #25
        "jump up", #26
        "make a phone call/answer phone", #27
        "playing with phone/tablet", #28
        "typing on a keyboard", #29
        "pointing to something with finger", #30
        "taking a selfie", #31
        "check time (from watch)", #32
        "rub two hands together", #33
        "nod head/bow", #34
        "shake head", #35
        "wipe face", #36
        "salute", #37
        "put palms together", #38
        "cross hands in front (say stop)", #39
        "sneeze/cough", #40
        "staggering", #41
        "falling", #42
        "touch head (headache)", #43
        "touch chest (stomachache/heart pain)", #44
        "touch back (backache)", #45
        "touch neck (neckache)", #46
        "nausea or vomiting condition", #47
        "use a fan (with hand or paper)/feeling warm", #48
        "punching/slapping other person", #49
        "kicking other person", #50
        "pushing other person", #51
        "pat on back of other person", #52
        "point finger at the other person", #53
        "hugging other person", #54
        "giving something to other person", #55
        "touch other person's pocket", #56
        "handshaking", #57
        "walking towards each other", #58
        "walking apart from each other" #59
    ]

    if not (0 <= label < 60):
        raise ValueError("Label must be in the range [0, 59].")

    return ntu60_classes[label]

# ---- 1. Load confusion matrix ----
folder = './workdir/ntu_angle/'
file_path = f"{folder}/confusion_matrix.csv"

cm = pd.read_csv(file_path, header=None).values  # shape (60, 60)

# ---- 2. Plot heatmap ----
plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap="Blues", cbar=True)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(f"{folder}/zzz_confusion_mat.png")

# ---- 3. Compute per-class performance ----
# True positives = diagonal
tp = np.diag(cm)

# Total samples per class (row-wise sum)
total_per_class = cm.sum(axis=1)

# Accuracy per class (recall)
class_accuracy = tp / total_per_class

# Handle division by zero if any class has no samples
class_accuracy = np.nan_to_num(class_accuracy)

# ---- 4. Rank classes (worst → best) ----
ranked_indices = np.argsort(class_accuracy)

file_path = f"{folder}/zz_class_ranking.txt"

with open(file_path, "w") as f:
    f.write("Class ranking (worst → best):\n\n")
    
    for rank, idx in enumerate(ranked_indices):
        f.write(
            f"Rank {rank+1:2d}: Class {idx:2d} | "
            f"Accuracy = {class_accuracy[idx]:.4f} | "
            f"Name: {ntu60_label_to_name(idx)}\n"
        )

print(f"Saved ranking to {file_path}")