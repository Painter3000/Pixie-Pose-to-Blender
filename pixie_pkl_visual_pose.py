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

# === Verbindungen (Skelett) ===
SMPLX_EDGES = [
    ("pelvis", "left_hip"), ("pelvis", "right_hip"), ("pelvis", "spine1"),
    ("left_hip", "left_knee"), ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_foot"), ("right_ankle", "right_foot"),
    ("spine1", "spine2"), ("spine2", "spine3"), ("spine3", "neck"), ("neck", "head"),
    ("neck", "left_collar"), ("neck", "right_collar"),
    ("left_collar", "left_shoulder"), ("right_collar", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
    ("head", "jaw"), ("jaw", "left_eye"), ("jaw", "right_eye")
]

# === Hilfsfunktionen ===
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def apply_body_cam(joints, body_cam):
    scale, tx, ty = body_cam
    translation = np.array([tx, ty, 0.0])
    return joints * scale + translation

def plot_pose(joints, joint_names, joint_sides, partbody_pose=None, body_cam_pos=None, title="SMPL-X Pose Analyse"):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Gelenkpunkte + Labels
    for i, (name, side) in enumerate(zip(joint_names, joint_sides)):
        x, y, z = joints[i]
        ax.scatter(x, y, z, c=COLOR_MAP[side], marker='o', s=50)
        ax.text(x, y, z, name, fontsize=8, color='black')

        # Raumrichtungspfeile (Z-Achse aus Rotationsmatrix)
        if partbody_pose is not None and i < len(partbody_pose):
            R = partbody_pose[i]
            z_dir = R[:, 2] * 0.05  # Skalierte Z-Achse
            ax.quiver(x, y, z, z_dir[0], z_dir[1], z_dir[2], color='gray', length=0.05, normalize=True)

    # Verbindungslinien (Knochen)
    name_to_index = {name: idx for idx, name in enumerate(joint_names)}
    for joint_a, joint_b in SMPLX_EDGES:
        if joint_a in name_to_index and joint_b in name_to_index:
            idx_a = name_to_index[joint_a]
            idx_b = name_to_index[joint_b]
            xa, ya, za = joints[idx_a]
            xb, yb, zb = joints[idx_b]

            # Seitenfarbe wählen (z. B. Seite von joint_a)
            side = joint_sides[idx_a]
            color = COLOR_MAP.get(side, "black")

            ax.plot([xa, xb], [ya, yb], [za, zb], color=color, linewidth=2)

    # body_cam als gelber Marker
    if body_cam_pos is not None:
        ax.scatter(*body_cam_pos, c='yellow', marker='^', s=80)
        ax.text(*body_cam_pos, "body_cam", fontsize=9, color='black')

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

# Gelenke laden
joints_raw = pred.get("smplx_kpt3d", None)
if joints_raw is None:
    raise ValueError("Keine 'smplx_kpt3d' Daten in prediction.pkl gefunden.")
joints = joints_raw.squeeze()

# Pose-Matrizen laden (optional für Pfeile)
partbody_pose = param.get("partbody_pose", None)
if partbody_pose is not None and partbody_pose.ndim == 4:
    partbody_pose = partbody_pose.squeeze(0)

# body_cam anwenden
body_cam_pos = None
if "body_cam" in param:
    body_cam = param["body_cam"].squeeze()
    joints = apply_body_cam(joints, body_cam)
    body_cam_pos = np.array([body_cam[1], body_cam[2], 0.0])  # tx, ty, z=0

# Plot mit Labels, Farben, Verbindungslinien & Pfeilen
plot_pose(joints[:25], SMPLX_NAMES, SMPLX_SIDES, partbody_pose=partbody_pose, body_cam_pos=body_cam_pos)
