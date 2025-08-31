import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Gelenk-Namen & Farbcodierung ===
SMPLX_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw",
    "left_eye", "right_eye"
]

SMPLX_SIDES = [
    "center", "left", "right", "center", "left", "right",
    "center", "left", "right", "center", "left", "right",
    "center", "left", "right", "center", "left", "right",
    "left", "right", "left", "right", "center",
    "left", "right"
]

COLOR_MAP = {"left": "blue", "right": "red", "center": "green"}

# === Hilfsfunktionen ===
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def apply_body_cam(joints, body_cam):
    scale, tx, ty = body_cam
    translation = np.array([tx, ty, 0.0])
    return joints * scale + translation

def plot_joints_with_labels(joints, joint_names, joint_sides, title="SMPL-X Pose"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, (name, side) in enumerate(zip(joint_names, joint_sides)):
        x, y, z = joints[i]
        ax.scatter(x, y, z, c=COLOR_MAP[side], marker='o', s=40)
        ax.text(x, y, z, name, fontsize=8, color='black')

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-70)
    plt.tight_layout()
    plt.show()

# === Hauptablauf ===
param_path = "./ari_param.pkl"
pred_path = "./ari_prediction.pkl"

param = load_pkl(param_path)
pred = load_pkl(pred_path)

joints_raw = pred.get("smplx_kpt3d", None)
if joints_raw is None:
    raise ValueError("Keine 'smplx_kpt3d' Daten in prediction.pkl gefunden.")

joints = joints_raw.squeeze()

# Optional: body_cam anwenden
if "body_cam" in param:
    joints = apply_body_cam(joints, param["body_cam"].squeeze())

# Plot der ersten 25 Gelenke mit Namen & Farben
plot_joints_with_labels(joints[:25], SMPLX_NAMES, SMPLX_SIDES)

