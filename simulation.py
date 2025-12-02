"""
Four-Finger Hand Kinematics - Adduction Motion Simulation
MECE4602 - Introduction to Robotics, Project 2
Authors: Jaisel Singh, Dilara Baysal, Laura Xing, Lorenzo De Sanctis
"""

import numpy as np
import sympy as sp
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
import matplotlib.animation as animation
import warnings
import os

warnings.filterwarnings('ignore')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

print("Setting up symbolic kinematics...")
theta1_s, theta2_s, theta3_s = sp.symbols('theta1 theta2 theta3', real=True)
L1_s, L2_s, L3_s = sp.symbols('L1 L2 L3', positive=True, real=True)
theta4_s = theta3_s / 2


def DH_matrix_symbolic(a, alpha, d, theta):
    ca, sa = sp.cos(alpha), sp.sin(alpha)
    ct, st = sp.cos(theta), sp.sin(theta)
    return sp.Matrix([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d],
        [0,   0,      0,     1]
    ])


A1_s = DH_matrix_symbolic(0, sp.pi/2, 0, theta1_s)
A2_s = DH_matrix_symbolic(L1_s, 0, 0, theta2_s)
A3_s = DH_matrix_symbolic(L2_s, 0, 0, theta3_s)
A4_s = DH_matrix_symbolic(L3_s, 0, 0, theta4_s)

T01_s = A1_s
T02_s = A1_s * A2_s
T03_s = T02_s * A3_s
T04_s = T03_s * A4_s

p_ee_s = T04_s[:3, 3]
R04_s = T04_s[:3, :3]
x4_s = R04_s[:, 0]

z0 = sp.Matrix([0, 0, 1])
z1 = T01_s[:3, 2]
z2 = T02_s[:3, 2]
z3 = T03_s[:3, 2]
o0 = sp.Matrix([0, 0, 0])
o1 = T01_s[:3, 3]
o2 = T02_s[:3, 3]
o3 = T03_s[:3, 3]
o4 = T04_s[:3, 3]


def cross(a, b):
    return sp.Matrix([a[1]*b[2] - a[2]*b[1],
                      a[2]*b[0] - a[0]*b[2],
                      a[0]*b[1] - a[1]*b[0]])


Jv1 = cross(z0, o4 - o0)
Jv2 = cross(z1, o4 - o1)
Jv3 = cross(z2, o4 - o2) + sp.Rational(1, 2) * cross(z3, o4 - o3)
Jv_geo = sp.simplify(sp.Matrix([[Jv1[0], Jv2[0], Jv3[0]],
                                [Jv1[1], Jv2[1], Jv3[1]],
                                [Jv1[2], Jv2[2], Jv3[2]]]))

vars_all = (theta1_s, theta2_s, theta3_s, L1_s, L2_s, L3_s)
p_ee_func = sp.lambdify(vars_all, p_ee_s, 'numpy')
x4_func = sp.lambdify(vars_all, x4_s, 'numpy')
R04_func = sp.lambdify(vars_all, R04_s, 'numpy')
Jv_func = sp.lambdify(vars_all, Jv_geo, 'numpy')
print("Ready.")


@dataclass
class FingerGeometry:
    name: str
    L1: float
    L2: float
    L3: float
    base_x: float
    base_y: float
    base_rot_z: float

    @property
    def total_length(self) -> float:
        return self.L1 + self.L2 + self.L3


@dataclass
class JointAngles:
    theta1: float
    theta2: float
    theta3: float

    @property
    def theta4(self) -> float:
        return 0.5 * self.theta3

    def as_array(self) -> np.ndarray:
        return np.array([self.theta1, self.theta2, self.theta3])


def Rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


def finger_FK_local(angles: JointAngles, geom: FingerGeometry):
    t1, t2, t3 = angles.theta1, angles.theta2, angles.theta3
    L1, L2, L3 = geom.L1, geom.L2, geom.L3
    pos = np.array(p_ee_func(t1, t2, t3, L1, L2, L3)).flatten()
    R = np.array(R04_func(t1, t2, t3, L1, L2, L3))
    distal = np.array(x4_func(t1, t2, t3, L1, L2, L3)).flatten()
    return pos, R, distal


def finger_FK_global(angles: JointAngles, geom: FingerGeometry):
    pos_local, R_local, distal_local = finger_FK_local(angles, geom)
    R_base = Rz(geom.base_rot_z)
    t_base = np.array([geom.base_x, geom.base_y, 0.0])
    return R_base @ pos_local + t_base, R_base @ R_local, R_base @ distal_local


def finger_IK_for_flexion(theta2_target: float, theta3_target: float,
                          theta1_val: float, geom: FingerGeometry) -> JointAngles:
    return JointAngles(theta1_val, theta2_target, theta3_target)


def get_all_joint_positions(angles: JointAngles, geom: FingerGeometry) -> np.ndarray:
    """Return all joint positions (MCP, PIP, DIP, fingertip) in the global frame."""
    t1, t2, t3 = angles.theta1, angles.theta2, angles.theta3
    L1, L2, L3 = geom.L1, geom.L2, geom.L3

    T01 = np.array(DH_matrix_symbolic(0, sp.pi/2, 0, t1)).astype(float)
    T12 = np.array(DH_matrix_symbolic(L1, 0, 0, t2)).astype(float)
    T23 = np.array(DH_matrix_symbolic(L2, 0, 0, t3)).astype(float)
    T34 = np.array(DH_matrix_symbolic(L3, 0, 0, 0.5*t3)).astype(float)

    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34

    joints_local = [
        np.array([0, 0, 0, 1.0]),
        T02 @ np.array([0, 0, 0, 1.0]),
        T03 @ np.array([0, 0, 0, 1.0]),
        T04 @ np.array([0, 0, 0, 1.0]),
    ]
    joints_local = np.stack(joints_local, axis=0)[:, :3]

    R_base = Rz(geom.base_rot_z)
    t_base = np.array([geom.base_x, geom.base_y, 0.0])

    joints_global = (R_base @ joints_local.T).T + t_base
    return joints_global


def finger_Jacobian_global(angles: JointAngles, geom: FingerGeometry) -> np.ndarray:
    t1, t2, t3 = angles.theta1, angles.theta2, angles.theta3
    L1, L2, L3 = geom.L1, geom.L2, geom.L3
    Jv = np.array(Jv_func(t1, t2, t3, L1, L2, L3), dtype=float)
    R_base = Rz(geom.base_rot_z)
    return R_base @ Jv


def compute_joint_torques(angles: JointAngles, geom: FingerGeometry, F_ee: np.ndarray) -> np.ndarray:
    Jv = finger_Jacobian_global(angles, geom)
    return Jv.T @ F_ee

# generate_adduction_trajectory with X/Z–aligned final fingertips
def generate_adduction_trajectory(fingers: List[FingerGeometry],
                                   n_frames: int,
                                   theta2_initial: float,
                                   theta2_final: float,
                                   theta3_initial: float,
                                   theta3_final: float,
                                   theta1_splay: Dict[str, float] = None) -> Dict:
    """
    Generate an adduction + flexion trajectory for all fingers.

    Key behavior: in the FINAL frame, all fingertips share a common
    X and Z coordinate (they lie on the same X/Z line) and differ only
    in Y.  We use the middle finger's nominal final pose as the
    reference and solve for per-finger flexion angles so that each
    fingertip matches the middle finger in X/Z.
    """
    # Default initial ab/adduction (splay) angles if none are provided.
    # These only affect the initial pose; all fingers adduct to 0 rad.
    if theta1_splay is None:
        theta1_splay = {
            'Index':  np.radians(15),
            'Middle': np.radians(0),
            'Ring':   np.radians(-10),
            'Pinky':  np.radians(-20),
        }

    results = {
        'n_frames': n_frames,
        'fingers': {}
    }

    t = np.linspace(0.0, 1.0, n_frames)

    # 1) Reference fingertip X/Z from the middle finger's final pose
    middle_geom = next((g for g in fingers if g.name == "Middle"), fingers[0])
    theta1_mid_initial = theta1_splay.get("Middle", 0.0)
    theta1_mid_final = theta1_mid_initial * 0.0  # -> 0 rad
    ref_angles = JointAngles(theta1_mid_final, theta2_final, theta3_final)
    ref_pos, _, _ = finger_FK_global(ref_angles, middle_geom)
    target_x, target_z = float(ref_pos[0]), float(ref_pos[2])

    # 2) Per-finger flexion targets so tips share (target_x, target_z)
    for geom in fingers:
        finger_data = {
            'angles': [],
            'positions': [],
            'theta1_values': [],
            'z_values': [],
            'valid': []
        }

        theta1_initial = theta1_splay.get(geom.name, 0.0)
        theta1_final = 0.0
        theta1_trajectory = theta1_initial * (1.0 - t)

        # Middle finger keeps the nominal flexion targets.
        if geom.name == "Middle":
            theta2_final_finger = theta2_final
            theta3_final_finger = theta3_final
        else:
            def obj(vars):
                t2, t3 = vars
                ang = JointAngles(theta1_final, t2, t3)
                pos, _, _ = finger_FK_global(ang, geom)
                return [pos[0] - target_x, pos[2] - target_z]

            sol, info, ier, msg = fsolve(
                obj,
                x0=[theta2_final, theta3_final],
                full_output=True,
                maxfev=2000
            )

            if ier != 1:
                # Fallback: use the nominal flexion if solve fails
                theta2_final_finger = theta2_final
                theta3_final_finger = theta3_final
            else:
                theta2_final_finger, theta3_final_finger = sol

        theta2_trajectory = theta2_initial + (theta2_final_finger - theta2_initial) * t
        theta3_trajectory = theta3_initial + (theta3_final_finger - theta3_initial) * t

        for frame_idx in range(n_frames):
            theta1_val = theta1_trajectory[frame_idx]
            theta2_val = theta2_trajectory[frame_idx]
            theta3_val = theta3_trajectory[frame_idx]

            angles = JointAngles(theta1_val, theta2_val, theta3_val)
            pos, R, distal = finger_FK_global(angles, geom)

            finger_data['angles'].append(angles)
            finger_data['positions'].append(pos)
            finger_data['theta1_values'].append(np.degrees(angles.theta1))
            finger_data['z_values'].append(pos[2])
            finger_data['valid'].append(True)

        results['fingers'][geom.name] = finger_data

    return results


def plot_angles_vs_frame(results: Dict, title: str) -> plt.Figure:
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    n_frames = results['n_frames']
    frames = np.arange(n_frames)
    colors = {'Index': '#1f77b4', 'Middle': '#d62728', 'Ring': '#2ca02c', 'Pinky': '#ff7f0e'}
    angle_names = ['θ₁ (MCP AbAd)', 'θ₂ (MCP Flex)', 'θ₃ (PIP)', 'θ₄ (DIP)']

    for i, ax in enumerate(axes):
        for name, data in results['fingers'].items():
            angles = data['angles']
            if i == 0:
                vals = [np.degrees(a.theta1) for a in angles]
            elif i == 1:
                vals = [np.degrees(a.theta2) for a in angles]
            elif i == 2:
                vals = [np.degrees(a.theta3) for a in angles]
            else:
                vals = [np.degrees(a.theta4) for a in angles]

            ax.plot(frames, vals, marker='o', label=name, color=colors[name])

        ax.set_ylabel(angle_names[i] + " (deg)")
        ax.grid(True)
        if i == 0:
            ax.legend()

    axes[-1].set_xlabel("Frame")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_torques_vs_frame(results: Dict, hand: List[FingerGeometry], F_ee: np.ndarray) -> plt.Figure:
    n_frames = results['n_frames']
    fingers = ['Index', 'Middle', 'Ring', 'Pinky']
    colors = {'Index': '#1f77b4', 'Middle': '#d62728', 'Ring': '#2ca02c', 'Pinky': '#ff7f0e'}

    tau1 = {name: [] for name in fingers}
    tau2 = {name: [] for name in fingers}
    tau3 = {name: [] for name in fingers}

    for frame_idx in range(n_frames):
        for name in fingers:
            geom = next(f for f in hand if f.name == name)
            angles = results['fingers'][name]['angles'][frame_idx]
            Jv = finger_Jacobian_global(angles, geom)
            tau = Jv.T @ F_ee
            tau1[name].append(tau[0])
            tau2[name].append(tau[1])
            tau3[name].append(tau[2])

    frames = np.arange(n_frames)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    for name in fingers:
        axes[0].plot(frames, tau1[name], label=name, color=colors[name])
        axes[1].plot(frames, tau2[name], label=name, color=colors[name])
        axes[2].plot(frames, tau3[name], label=name, color=colors[name])

    axes[0].set_ylabel("τ₁ (MCP AbAd) (N·m)")
    axes[1].set_ylabel("τ₂ (MCP Flex) (N·m)")
    axes[2].set_ylabel("τ₃ (PIP) (N·m)")
    axes[2].set_xlabel("Frame")
    axes[0].set_title(f"Joint Torques During Adduction (F={F_ee})")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    return fig


def plot_hand_3d(hand: List[FingerGeometry], angles_list: List[JointAngles], title: str) -> plt.Figure:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    colors = {'Index': '#1f77b4', 'Middle': '#d62728', 'Ring': '#2ca02c', 'Pinky': '#ff7f0e'}

    palm_y = np.linspace(-0.3, 0.5, 5)
    ax.plot(np.zeros_like(palm_y), palm_y, np.zeros_like(palm_y), 'k-', linewidth=3, alpha=0.7)

    for geom, angles in zip(hand, angles_list):
        joints = get_all_joint_positions(angles, geom)
        ax.plot(joints[:, 0], joints[:, 1], joints[:, 2], 'o-',
                color=colors.get(geom.name, 'gray'), linewidth=2, markersize=6, label=geom.name)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim([-0.2, 0.9]); ax.set_ylim([-0.4, 0.6]); ax.set_zlim([-0.3, 0.6])
    ax.legend()
    return fig


def plot_jacobian_analysis(hand: List[FingerGeometry], angles_list: List[JointAngles]) -> plt.Figure:
    fingers = [geom.name for geom in hand]
    conds = []
    manips = []

    for geom, angles in zip(hand, angles_list):
        Jv = finger_Jacobian_global(angles, geom)
        cond = np.linalg.cond(Jv)
        manip = np.sqrt(np.linalg.det(Jv @ Jv.T))
        conds.append(cond)
        manips.append(manip)

    x = np.arange(len(fingers))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(x, conds, width)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(fingers)
    axes[0].set_ylabel("Condition Number")
    axes[0].set_title("Jacobian Condition Number\n(lower = better)")

    axes[1].bar(x, manips, width)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(fingers)
    axes[1].set_ylabel("Manipulability")
    axes[1].set_title("Manipulability Index\n(higher = better)")

    fig.tight_layout()
    return fig


def create_adduction_animation(hand: List[FingerGeometry], results: Dict, interval: int = 150, fps: int = 5,
                               save_path: Optional[str] = None) -> animation.FuncAnimation:
    n_frames = results['n_frames']
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    colors = {'Index': '#1f77b4', 'Middle': '#d62728', 'Ring': '#2ca02c', 'Pinky': '#ff7f0e'}

    palm_y = np.linspace(-0.3, 0.5, 5)

    def init():
        ax.plot(np.zeros_like(palm_y), palm_y, np.zeros_like(palm_y), 'k-', linewidth=3, alpha=0.7)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_xlim([-0.2, 0.9]); ax.set_ylim([-0.4, 0.6]); ax.set_zlim([-0.3, 0.6])
        return []

    def update(frame_idx):
        ax.cla()
        ax.plot(np.zeros_like(palm_y), palm_y, np.zeros_like(palm_y), 'k-', linewidth=3, alpha=0.7)
        for geom in hand:
            data = results['fingers'][geom.name]
            angles = data['angles'][frame_idx]
            joints = get_all_joint_positions(angles, geom)
            ax.plot(joints[:, 0], joints[:, 1], joints[:, 2], 'o-',
                    color=colors.get(geom.name, 'gray'), linewidth=2, markersize=6)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_xlim([-0.2, 0.9]); ax.set_ylim([-0.4, 0.6]); ax.set_zlim([-0.3, 0.6])
        ax.set_title(f"Adduction Motion - Frame {frame_idx+1}/{n_frames}", fontsize=12)
        return []

    anim = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init,
                                   interval=interval, blit=False)

    if save_path is not None:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='pillow', fps=fps)
        print("Animation saved!")

    return anim


def create_hand_from_measurements(index_lengths, middle_lengths, ring_lengths, pinky_lengths,
                                   finger_spacing=(-2.0, 0.0, 2.0, 3.8),
                                   splay_angles=(0, 0, 0, 0)):
    scale = 1.0 / sum(middle_lengths)
    return [
        FingerGeometry("Index", index_lengths[0]*scale, index_lengths[1]*scale, index_lengths[2]*scale,
                       0.0, finger_spacing[0]*scale, np.radians(splay_angles[0])),
        FingerGeometry("Middle", middle_lengths[0]*scale, middle_lengths[1]*scale, middle_lengths[2]*scale,
                       0.0, finger_spacing[1]*scale, np.radians(splay_angles[1])),
        FingerGeometry("Ring", ring_lengths[0]*scale, ring_lengths[1]*scale, ring_lengths[2]*scale,
                       0.0, finger_spacing[2]*scale, np.radians(splay_angles[2])),
        FingerGeometry("Pinky", pinky_lengths[0]*scale, pinky_lengths[1]*scale, pinky_lengths[2]*scale,
                       0.0, finger_spacing[3]*scale, np.radians(splay_angles[3])),
    ]


if __name__ == "__main__":
    print("=" * 70)
    print("  ADDUCTION MOTION SIMULATION")
    print("  Extended (splayed) → Contracted (together)")
    print("=" * 70)

    hand = create_hand_from_measurements(
        index_lengths=(4.0, 2.5, 1.8),
        middle_lengths=(4.5, 3.0, 2.0),
        ring_lengths=(4.2, 2.8, 1.9),
        pinky_lengths=(3.0, 2.0, 1.5),
        finger_spacing=(-2.0, 0.0, 2.0, 3.8),
        splay_angles=(0, 0, 0, 0)
    )

    print("\nHand Configuration:")
    for f in hand:
        print(f"  {f.name:8s}: L={f.total_length:.3f}, base_y={f.base_y:.3f}")

    print("\nGenerating adduction trajectory...")
    n_frames = 25

    results = generate_adduction_trajectory(
        hand,
        n_frames=n_frames,
        theta2_initial=np.radians(0),
        theta2_final=np.radians(70),
        theta3_initial=np.radians(5),
        theta3_final=np.radians(90),
        theta1_splay={
            'Index': np.radians(-15),
            'Middle': np.radians(0),
            'Ring': np.radians(12),
            'Pinky': np.radians(25)
        }
    )

    print("\nTrajectory Summary:")
    for name, data in results['fingers'].items():
        valid = sum(data['valid'])
        theta1_initial = data['theta1_values'][0]
        theta1_final = data['theta1_values'][-1]
        a_init = data['angles'][0]
        a_final = data['angles'][-1]
        print(f"  {name:8s}: θ₁: {theta1_initial:6.1f}° → {theta1_final:6.1f}°, "
              f"θ₂: {np.degrees(a_init.theta2):5.1f}° → {np.degrees(a_final.theta2):5.1f}°, "
              f"θ₃: {np.degrees(a_init.theta3):5.1f}° → {np.degrees(a_final.theta3):5.1f}°")

    F_down = np.array([0.0, 0.0, -1.0])

    print("\nGenerating plots...")

    initial_angles = [data['angles'][0] for data in results['fingers'].values()]
    fig1 = plot_angles_vs_frame(results, "Joint Angles During Adduction Motion")
    fig1.savefig(os.path.join(OUTPUT_DIR, 'angles_vs_frame.png'), dpi=150, bbox_inches='tight')
    print("  Saved: angles_vs_frame.png")

    fig2 = plot_torques_vs_frame(results, hand, F_down)
    fig2.savefig(os.path.join(OUTPUT_DIR, 'torques_vs_frame.png'), dpi=150, bbox_inches='tight')
    print("  Saved: torques_vs_frame.png")

    fig3 = plot_hand_3d(hand, initial_angles, "INITIAL: Extended & Splayed (θ₁≠0, θ₂≈0°, θ₃≈5°)")
    fig3.savefig(os.path.join(OUTPUT_DIR, 'hand_initial.png'), dpi=150, bbox_inches='tight')
    print("  Saved: hand_initial.png")

    final_angles = [data['angles'][-1] for data in results['fingers'].values()]
    fig4 = plot_hand_3d(hand, final_angles, "FINAL: Contracted & Adducted (θ₁=0, θ₂≈70°, θ₃≈90°)")
    fig4.savefig(os.path.join(OUTPUT_DIR, 'hand_final.png'), dpi=150, bbox_inches='tight')
    print("  Saved: hand_final.png")

    fig5 = plot_jacobian_analysis(hand, final_angles)
    fig5.savefig(os.path.join(OUTPUT_DIR, 'jacobian_final.png'), dpi=150, bbox_inches='tight')
    print("  Saved: jacobian_final.png")

    fig6, axes = plt.subplots(1, 3, figsize=(16, 5), subplot_kw={'projection': '3d'})
    colors = {'Index': '#1f77b4', 'Middle': '#d62728', 'Ring': '#2ca02c', 'Pinky': '#ff7f0e'}

    for ax, idx, label in zip(axes, [0, n_frames//2, n_frames-1],
                              ['INITIAL (Extended)', 'MID', 'FINAL (Contracted)']):
        for geom in hand:
            data = results['fingers'][geom.name]
            angles = data['angles'][idx]
            if angles is None:
                continue
            joints = get_all_joint_positions(angles, geom)
            ax.plot(joints[:, 0], joints[:, 1], joints[:, 2], 'o-',
                    color=colors.get(geom.name, 'gray'), linewidth=2, markersize=6)

        palm_y = np.linspace(-0.3, 0.5, 5)
        ax.plot(np.zeros_like(palm_y), palm_y, np.zeros_like(palm_y), 'k-', linewidth=3, alpha=0.7)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlim([-0.2, 0.9]); ax.set_ylim([-0.4, 0.6]); ax.set_zlim([-0.3, 0.6])

    fig6.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig6.savefig(os.path.join(OUTPUT_DIR, 'trajectory_sequence.png'), dpi=150, bbox_inches='tight')
    print("  Saved: trajectory_sequence.png")

    print("\nCreating animation...")
    anim_path = os.path.join(OUTPUT_DIR, 'adduction_animation.gif')
    anim = create_adduction_animation(hand, results, interval=150, fps=5, save_path=anim_path)
    print(f"  Saved animation: {anim_path}")

    print("\nSample Joint Angles (deg) at INIT, MID, FINAL:")
    print(f"{'Frame':>6} | {'Finger':>8} | {'θ₁':>8} | {'θ₂':>8} | {'θ₃':>8} | {'θ₄':>8} |")
    print("-" * 65)

    for idx, label in [(0, 'INIT'), (n_frames//2, 'MID'), (n_frames-1, 'FINAL')]:
        for name in ['Index', 'Middle', 'Ring', 'Pinky']:
            data = results['fingers'][name]
            a = data['angles'][idx]
            print(f"{label:>6} | {name:>8} | {np.degrees(a.theta1):8.2f} | "
                  f"{np.degrees(a.theta2):8.2f} | {np.degrees(a.theta3):8.2f} | "
                  f"{np.degrees(a.theta4):8.2f} |")
        print("-" * 65)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    plt.show()
