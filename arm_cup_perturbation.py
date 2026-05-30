"""
arm_cup_perturbation.py

Point-to-point cup trajectory (left → right, minimum-jerk) with a single
mechanical perturbation, matching the "Cup-and-Ball + Perturbation" condition
(purple line in the reference figure).

Target profiles
---------------
  Cup velocity  : bell-shaped (minimum-jerk), peak ≈ 0.35 m/s at t ≈ T/2
                  with a sharp velocity dip at t_pert due to the perturbation
  Interaction F : ≈ 0 during smooth motion; Gaussian spike at t_pert (≈ 12 N)

Physical model
--------------
  The perturbation is a brief external force F_pert(t) applied AGAINST motion.
  It modifies cup acceleration via Newton's 2nd law:

      a(t) = a_min_jerk(t)  −  F_pert(t) / m_eff

  where m_eff = effective endpoint inertia (arm + cup + ball ≈ 2 kg).

  The interaction force measured at the cup handle ≈ F_pert(t), because the
  hand must push forward to resist the perturbation (Newton's 3rd law).

IK
--
  Same two-DOF approach as arm_linear_trajectory.py:
  elv_angle + elbow_flexion solved via L-BFGS-B; all other DOFs locked.

Outputs  (demo_output/)
-----------------------
  cup_task_trajectory_perturb.csv / .mot
  cup_task_velocity_force_perturb.png   – 2-panel plot (velocity + force)
  cup_task_endpoint_path_perturb.png    – 3-D endpoint path

Usage
-----
  python arm_cup_perturbation.py
  python arm_cup_perturbation.py --t-pert 1.0 --f-pert 15 --sigma 0.03
"""

import os
import sys
import argparse

_simbody_dir = os.path.join(sys.prefix, "libexec", "simbody")
if _simbody_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _simbody_dir + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import pandas as pd
import opensim as osim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize as sp_minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_cup_task import MODEL_PATH, DEMO_OUTPUT_DIR, ACTIVE_DOFS, NEUTRAL, FS, write_mot, run_msk, MS_LABELS
from config import (
    T_MOVE, D_MOVE, M_EFF,
    T_PERT, F_PERT, SIGMA_PERT, V_DIP_FRAC,
    F_NEG, DT_NEG, SIG_NEG,
    ELV_ANGLE_BOUNDS, ELBOW_FLEX_BOUNDS,
    HAND_BODY, HAND_LOCAL_PT,
)
import os as _os
SCRIPT_NAME = _os.path.splitext(_os.path.basename(__file__))[0]
OUTPUT_DIR  = _os.path.join(DEMO_OUTPUT_DIR, SCRIPT_NAME)
_os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================================================
# Parameters  (all tunable via CLI)
# ===========================================================================
TAG        = "perturb"

# Movement, perturbation, force-profile, IK-bound, and hand constants are
# imported from config.py (single source of truth). Script-specific only:

# ── IK ────────────────────────────────────────────────────────────────────
IK_FS      = 20      # Hz  – coarse IK grid
VISUALIZE  = False

# ── Locked DOFs (derived from NEUTRAL imported via arm_cup_task) ───────────
LOCKED = {
    "shoulder_elv": NEUTRAL["shoulder_elv"],
    "shoulder_rot": NEUTRAL["shoulder_rot"],
    "pro_sup":      NEUTRAL["pro_sup"],
    "deviation":    NEUTRAL["deviation"],
    "flexion":      NEUTRAL["flexion"],
}

# ===========================================================================
# 1-D trajectory functions
# ===========================================================================

def _mj_pos(t: np.ndarray) -> np.ndarray:
    """Minimum-jerk position  x(t) ∈ [0, D_MOVE]."""
    tau = np.clip(t / T_MOVE, 0.0, 1.0)
    return D_MOVE * (10*tau**3 - 15*tau**4 + 6*tau**5)

def _mj_vel(t: np.ndarray) -> np.ndarray:
    """Minimum-jerk velocity  ẋ(t)."""
    tau = np.clip(t / T_MOVE, 0.0, 1.0)
    return (D_MOVE / T_MOVE) * (30*tau**2 - 60*tau**3 + 30*tau**4)

def _mj_acc(t: np.ndarray) -> np.ndarray:
    """Minimum-jerk acceleration  ẍ(t)."""
    tau = np.clip(t / T_MOVE, 0.0, 1.0)
    return (D_MOVE / T_MOVE**2) * (60*tau - 180*tau**2 + 120*tau**3)

def _f_pert(t: np.ndarray) -> np.ndarray:
    """Gaussian perturbation force [N], opposing motion (positive = resisting)."""
    return F_PERT * np.exp(-0.5 * ((t - T_PERT) / SIGMA_PERT)**2)

def build_1d_trajectory() -> dict:
    """
    Prescribe velocity = min-jerk baseline MINUS a Gaussian velocity dip.

    The dip depth is V_DIP_FRAC × v(T_PERT), so the arm still travels
    ≈ D_MOVE − V_DIP_FRAC × v(T_PERT) × SIGMA_PERT × √(2π)  ≈ 0.27 m
    (most of the intended displacement is preserved).

    The interaction force F_int is prescribed separately as a Gaussian
    spike at T_PERT — it represents the reactive force the hand exerts
    on the cup handle to oppose the braking perturbation.

    Returns dict: t, pos, vel, acc, F_pert, F_int
    """
    dt  = 1.0 / FS
    t   = np.arange(0.0, T_MOVE + dt / 2, dt)

    # Baseline minimum-jerk velocity
    v_mj = _mj_vel(t)

    # Velocity dip: Gaussian subtraction scaled to V_DIP_FRAC × v at T_PERT
    v_at_pert = float(_mj_vel(np.array([T_PERT]))[0])
    v_dip     = V_DIP_FRAC * v_at_pert * np.exp(-0.5 * ((t - T_PERT) / SIGMA_PERT)**2)
    vel       = v_mj - v_dip

    # Position: integrate prescribed velocity
    pos = np.concatenate([[0.0], cumulative_trapezoid(vel, t)])

    # Acceleration: numerical derivative (for reference / ID)
    acc = np.gradient(vel, t)

    # Interaction force: three-component model matching purple-line profile
    #   (1) slow baseline from inertial coupling with min-jerk acceleration
    F_base  = M_EFF * _mj_acc(t)
    #   (2) sharp spike at perturbation (braking reaction)
    F_spike = F_PERT * np.exp(-0.5 * ((t - T_PERT) / SIGMA_PERT)**2)
    #   (3) negative undershoot after spike (rebound / over-deceleration)
    F_under = F_NEG  * np.exp(-0.5 * ((t - (T_PERT + DT_NEG)) / SIG_NEG)**2)
    F_int   = F_base + F_spike - F_under

    print(f"  Min-jerk peak vel    : {v_mj.max():.3f} m/s  at t={t[v_mj.argmax()]:.3f} s")
    print(f"  Velocity at t_pert   : {v_at_pert:.3f} m/s")
    print(f"  Velocity dip depth   : {v_dip.max():.3f} m/s  ({V_DIP_FRAC*100:.0f}% of v at T_PERT)")
    print(f"  Velocity after dip   : {vel[np.argmax(v_dip)]:.3f} m/s")
    print(f"  Peak F_int (spike)   : {F_spike.max():.1f} N")
    print(f"  Min F_int (undershoot): {F_int.min():.1f} N")
    print(f"  Total displacement   : {pos[-1]*100:.1f} cm  (nominal {D_MOVE*100:.0f} cm)")

    return {"t": t, "pos": pos, "vel": vel, "acc": acc,
            "v_dip": v_dip,    # velocity dip (m/s) — kept for diagnostics
            "F_pert": F_spike, # perturbation force spike (N) — used for mask detection
            "F_int": F_int}


# ===========================================================================
# OpenSim FK / IK
# ===========================================================================

def _init_model():
    m = osim.Model(MODEL_PATH)
    m.setUseVisualizer(VISUALIZE)
    s = m.initSystem()
    for dof, val in LOCKED.items():
        m.updCoordinateSet().get(dof).setValue(s, val, False)
    m.assemble(s)
    return m, s


def _hand_pos(model, state, q: np.ndarray) -> np.ndarray:
    """FK: set (elv_angle, elbow_flexion) = q, return hand position in ground."""
    model.updCoordinateSet().get("elv_angle").setValue(state, q[0], False)
    model.updCoordinateSet().get("elbow_flexion").setValue(state, q[1], False)
    model.realizePosition(state)
    p = model.getBodySet().get(HAND_BODY).findStationLocationInGround(
        state, osim.Vec3(*HAND_LOCAL_PT))
    return np.array([p[0], p[1], p[2]])


def _ik_cost(q, model, state, target):
    return float(np.sum((_hand_pos(model, state, q) - target)**2))


def _solve_ik(model, state, target, q0):
    res = sp_minimize(
        _ik_cost, x0=q0, args=(model, state, target),
        method="L-BFGS-B",
        bounds=[ELV_ANGLE_BOUNDS, ELBOW_FLEX_BOUNDS],
        options={"ftol": 1e-14, "gtol": 1e-8, "maxiter": 300},
    )
    return res.x


def _line_geometry(model, state) -> tuple:
    """
    Return (center, direction) of the left-right sweep line.
    Identical to arm_linear_trajectory.py: sample FK at elv_angle ± 10°.
    """
    q0   = np.array([NEUTRAL["elv_angle"], NEUTRAL["elbow_flexion"]])
    q_lo = np.array([NEUTRAL["elv_angle"] - np.radians(10.0), NEUTRAL["elbow_flexion"]])
    q_hi = np.array([NEUTRAL["elv_angle"] + np.radians(10.0), NEUTRAL["elbow_flexion"]])
    center    = _hand_pos(model, state, q0)
    diff      = _hand_pos(model, state, q_hi) - _hand_pos(model, state, q_lo)
    direction = diff / np.linalg.norm(diff)
    print(f"  Hand centre   : {np.round(center, 3)}")
    print(f"  Line direction: {np.round(direction, 3)}")
    return center, direction


# ===========================================================================
# Trajectory builder
# ===========================================================================

def build_perturbed_trajectory(model, state) -> tuple:
    """
    Map the 1-D perturbed cup trajectory to 3-D hand endpoint targets
    and solve IK (coarse 20 Hz grid → cubic spline → 100 Hz).

    Returns
    -------
    traj       DataFrame  – all ACTIVE_DOFS at FS Hz
    cup_1d     dict       – 1-D signals (t, pos, vel, acc, F_int)
    center     (3,)
    direction  (3,)
    targets_3d (N, 3)
    actuals_3d (N, 3)
    """
    center, direction = _line_geometry(model, state)
    cup_1d = build_1d_trajectory()

    t_full  = cup_1d["t"]
    pos_1d  = cup_1d["pos"]

    # Centre the motion: start at left (−D/2), end near right (+D/2)
    pos_offset  = pos_1d - D_MOVE / 2.0
    targets_3d  = (center[np.newaxis, :]
                   + pos_offset[:, np.newaxis] * direction[np.newaxis, :])

    # ── coarse IK grid ──────────────────────────────────────────────────
    n_c     = int(T_MOVE * IK_FS) + 1
    t_c     = np.linspace(0.0, T_MOVE, n_c)
    tgt_c   = np.stack([interp1d(t_full, targets_3d[:, k])(t_c) for k in range(3)],
                       axis=1)

    q        = np.array([NEUTRAL["elv_angle"], NEUTRAL["elbow_flexion"]])
    q_coarse = np.zeros((n_c, 2))
    for i, tgt in enumerate(tqdm(tgt_c, total=n_c, desc="IK – coarse", ncols=80)):
        q = _solve_ik(model, state, tgt, q0=q)
        q_coarse[i] = q

    # ── spline → 100 Hz ─────────────────────────────────────────────────
    cs_elv  = CubicSpline(t_c, q_coarse[:, 0])
    cs_flex = CubicSpline(t_c, q_coarse[:, 1])

    elv_full  = np.clip(cs_elv(t_full),  *ELV_ANGLE_BOUNDS)
    flex_full = np.clip(cs_flex(t_full), *ELBOW_FLEX_BOUNDS)

    # ── FK verification ─────────────────────────────────────────────────
    actuals_3d = np.zeros((len(t_full), 3))
    for i, (a, f) in enumerate(zip(elv_full, flex_full)):
        actuals_3d[i] = _hand_pos(model, state, np.array([a, f]))

    # ── DataFrame ───────────────────────────────────────────────────────
    records = []
    for t_i, a, f in zip(t_full, elv_full, flex_full):
        row = {"time": t_i, "elv_angle": a, "elbow_flexion": f}
        row.update(LOCKED)
        records.append(row)

    traj = pd.DataFrame(records)[["time"] + ACTIVE_DOFS]
    return traj, cup_1d, center, direction, targets_3d, actuals_3d


# ===========================================================================
# Plots
# ===========================================================================

def plot_velocity_force(cup_1d: dict, save_path: str):
    """
    2-panel plot mirroring columns C of the reference figure:
      top    – cup velocity [m/s]
      bottom – interaction force [N]
    Purple line colour #9B30FF to match the reference.
    """
    t     = cup_1d["t"]
    vel   = cup_1d["vel"]
    F_int = cup_1d["F_int"]

    PURPLE = "#9B30FF"

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    # ── top: velocity ────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, vel, color=PURPLE, linewidth=2.2, label="Cup velocity")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.axvline(T_PERT, color="red", linewidth=0.8, linestyle=":",
               alpha=0.7, label=f"Perturbation  t={T_PERT:.2f} s")
    ax.set_ylabel("Cup Velocity (m/s)", fontsize=11)
    ax.set_ylim(-0.08, 0.55)
    ax.set_xlim(0, T_MOVE)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title("Cup-and-Ball + Perturbation", fontsize=11)

    # ── inset: velocity zoom around perturbation ─────────────────────────
    ax_in = ax.inset_axes([0.52, 0.55, 0.38, 0.38])
    win = 0.25  # ± window around T_PERT
    mask = (t >= T_PERT - win) & (t <= T_PERT + win)
    ax_in.plot(t[mask], vel[mask], color=PURPLE, linewidth=1.5)
    ax_in.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_in.axvline(T_PERT, color="red", linewidth=0.7, linestyle=":")
    ax_in.set_xlim(T_PERT - win, T_PERT + win)
    ax_in.tick_params(labelsize=7)
    ax_in.set_title("zoom", fontsize=7)
    ax_in.grid(True, alpha=0.25)
    ax.indicate_inset_zoom(ax_in, edgecolor="black", linewidth=0.8)

    # ── bottom: interaction force ─────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, F_int, color=PURPLE, linewidth=2.2, label="Interaction force")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.axvline(T_PERT, color="red", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_ylabel("Interaction Force (N)", fontsize=11)
    ax.set_xlabel("Time (s)", fontsize=11)
    f_min = min(F_int.min(), 0) * 1.4
    ax.set_ylim(f_min, F_PERT * 1.3)
    ax.set_xlim(0, T_MOVE)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── inset: force zoom (wider window to show undershoot) ──────────────
    win_f  = 0.45
    mask_f = (t >= T_PERT - 0.10) & (t <= T_PERT + win_f)
    ax_in2 = ax.inset_axes([0.45, 0.35, 0.50, 0.55])
    ax_in2.plot(t[mask_f], F_int[mask_f], color=PURPLE, linewidth=1.5)
    ax_in2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_in2.axvline(T_PERT, color="red", linewidth=0.7, linestyle=":")
    ax_in2.set_xlim(T_PERT - 0.10, T_PERT + win_f)
    ax_in2.tick_params(labelsize=7)
    ax_in2.set_title("zoom", fontsize=7)
    ax_in2.grid(True, alpha=0.25)
    ax.indicate_inset_zoom(ax_in2, edgecolor="black", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_endpoint_path(center, direction, targets_3d, actuals_3d, save_path):
    fig = plt.figure(figsize=(9, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot(targets_3d[:, 0], targets_3d[:, 1], targets_3d[:, 2],
            "r--", linewidth=1.5, label="Target (min-jerk + perturb)")
    ax.plot(actuals_3d[:, 0], actuals_3d[:, 1], actuals_3d[:, 2],
            "b-",  linewidth=1.5, label="IK achieved")
    ax.scatter(*center, color="k", s=60, zorder=5, label="Neutral centre")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("Hand endpoint: perturbed trajectory")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_muscle_lengths(ms_lens: pd.DataFrame, save_path_norm: str, save_path_delta: str):
    """
    Two figures:
      1. Normalised fiber lengths  ℓ̃_f(t)  for all 14 muscles.
      2. Fiber length change  Δℓ̃_f(t) = ℓ̃_f(t) − ℓ̃_f(t=0).

    Red dashed vertical line marks T_PERT.
    Muscles are grouped by function: flexors (blue), extensors (red),
    shoulder (green), wrist (orange).
    """
    t = ms_lens["time"].values

    GROUPS = {
        "Elbow flexors":  (["BIClong", "BICshort", "BRA", "BRD"],  "#1f77b4"),
        "Elbow extensors":(["TRIlong", "TRIlat", "TRImed"],          "#d62728"),
        "Shoulder":       (["DELT1", "DELT2", "DELT3"],              "#2ca02c"),
        "Wrist":          (["ECRL", "ECRB", "FCR", "FCU"],           "#ff7f0e"),
    }

    for save_path, ylabel, title, transform in [
        (save_path_norm,  "Normalised fiber length (ℓ/ℓ₀)",
         "Muscle Fiber Lengths vs Time",
         lambda x, x0: x),
        (save_path_delta, "Fiber length change  Δ(ℓ/ℓ₀)",
         "Muscle Fiber Length Change vs Time",
         lambda x, x0: x - x0),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        axes = axes.flatten()

        for ax, (group_name, (muscles, base_col)) in zip(axes, GROUPS.items()):
            import matplotlib.cm as cm
            n_m = len(muscles)
            colors = [matplotlib.colormaps["tab10"](i) for i in range(n_m)]
            for ms, col in zip(muscles, colors):
                if ms not in ms_lens.columns:
                    continue
                y = ms_lens[ms].values
                y0 = y[0]
                ax.plot(t, transform(y, y0), linewidth=1.6, label=ms, color=col)
            ax.axhline(0 if "Change" in title else 1.0,
                       color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
            ax.axvline(T_PERT, color="red", linewidth=0.9, linestyle=":",
                       alpha=0.7, label=f"T_pert={T_PERT}s")
            ax.set_title(group_name, fontsize=10, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.legend(fontsize=7, ncol=2, loc="best")
            ax.grid(True, alpha=0.25)

        fig.suptitle(title, fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {save_path}")


def plot_joint_angles(traj: pd.DataFrame, save_path: str):
    """Plot all joint angles vs time."""
    t = traj["time"].values
    
    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, dof in enumerate(ACTIVE_DOFS):
        ax = axes[idx]
        q = traj[dof].values
        ax.plot(t, np.degrees(q), color="#0066CC", linewidth=1.5)
        ax.axhline(np.degrees(NEUTRAL[dof]), color="gray", 
                   linewidth=0.8, linestyle="--", alpha=0.6, label="Neutral")
        ax.axvline(T_PERT, color="red", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel(f"{dof} (deg)", fontsize=9)
        ax.set_title(f"{dof}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    
    # Hide the extra subplot
    axes[-1].set_visible(False)
    
    fig.suptitle("Joint Angles vs Time", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_joint_velocities(traj: pd.DataFrame, save_path: str):
    """Plot joint angle velocities (derivatives) vs time."""
    t = traj["time"].values
    dt = t[1] - t[0]
    
    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, dof in enumerate(ACTIVE_DOFS):
        ax = axes[idx]
        q = traj[dof].values
        # Compute velocity via gradient
        qdot = np.gradient(q, dt)
        ax.plot(t, np.degrees(qdot), color="#FF6600", linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(T_PERT, color="red", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel(f"d{dof}/dt (deg/s)", fontsize=9)
        ax.set_title(f"{dof} velocity", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
    
    # Hide the extra subplot
    axes[-1].set_visible(False)
    
    fig.suptitle("Joint Angle Velocities vs Time", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ===========================================================================
# Entry point
# ===========================================================================

# ===========================================================================
# Class wrapper — ArmCupPerturbation
# ===========================================================================
class ArmCupPerturbation:
    """Object-oriented driver for the perturbed cup trajectory pipeline.

    Wraps build_perturbed_trajectory + plotting into a single class. CLI
    overrides applied by ``run()`` mutate the module-level globals so that
    downstream imports (e.g. arm_inverse_dynamics) see consistent values.
    """

    def __init__(self,
                 t_pert: float = None,  f_pert: float = None,
                 sigma:  float = None,  v_dip:  float = None,
                 m_eff:  float = None,  d_move: float = None,
                 f_neg:  float = None,  dt_neg: float = None,
                 sig_neg: float = None,
                 output_dir: str = OUTPUT_DIR,
                 tag: str = TAG):
        # Apply overrides to module globals (kept for back-compat with
        # any caller that imports T_PERT, F_PERT, … directly).
        global T_PERT, F_PERT, SIGMA_PERT, V_DIP_FRAC, M_EFF, D_MOVE
        global F_NEG, DT_NEG, SIG_NEG
        if t_pert  is not None: T_PERT     = t_pert
        if f_pert  is not None: F_PERT     = f_pert
        if sigma   is not None: SIGMA_PERT = sigma
        if v_dip   is not None: V_DIP_FRAC = v_dip
        if m_eff   is not None: M_EFF      = m_eff
        if d_move  is not None: D_MOVE     = d_move
        if f_neg   is not None: F_NEG      = f_neg
        if dt_neg  is not None: DT_NEG     = dt_neg
        if sig_neg is not None: SIG_NEG    = sig_neg

        self.output_dir = output_dir
        self.tag        = tag
        os.makedirs(self.output_dir, exist_ok=True)

        # populated lazily
        self.model = None;     self.state    = None
        self.traj = None;      self.cup_1d   = None
        self.center = None;    self.direction = None
        self.targets_3d = None; self.actuals_3d = None
        self.ms_lens = None

    # ── Pipeline steps ────────────────────────────────────────────────────
    def init_model(self):
        self.model, self.state = _init_model()
        return self.model, self.state

    def build_trajectory(self):
        if self.model is None:
            self.init_model()
        (self.traj, self.cup_1d, self.center, self.direction,
         self.targets_3d, self.actuals_3d) = build_perturbed_trajectory(
            self.model, self.state)
        return self.traj

    def save_trajectory(self):
        traj_csv = os.path.join(self.output_dir, f"cup_task_trajectory_{self.tag}.csv")
        self.traj.to_csv(traj_csv, index=False)
        print(f"Trajectory : {traj_csv}  ({len(self.traj)} frames @ {FS} Hz)")

        mot_path = os.path.join(self.output_dir, f"cup_task_trajectory_{self.tag}.mot")
        write_mot(self.traj, mot_path)

        sig_csv = os.path.join(self.output_dir, f"cup_task_signals_{self.tag}.csv")
        pd.DataFrame({
            "time":   self.cup_1d["t"],
            "pos":    self.cup_1d["pos"],
            "vel":    self.cup_1d["vel"],
            "acc":    self.cup_1d["acc"],
            "v_dip":  self.cup_1d["v_dip"],   # velocity dip (m/s)
            "F_pert": self.cup_1d["F_pert"],  # perturbation force spike (N)
            "F_int":  self.cup_1d["F_int"],
        }).to_csv(sig_csv, index=False)
        print(f"Signals    : {sig_csv}")
        return {"traj_csv": traj_csv, "mot": mot_path, "sig_csv": sig_csv}

    def plot_all(self):
        plot_velocity_force(self.cup_1d,
            os.path.join(self.output_dir, f"cup_task_velocity_force_{self.tag}.png"))
        plot_endpoint_path(self.center, self.direction, self.targets_3d, self.actuals_3d,
            os.path.join(self.output_dir, f"cup_task_endpoint_path_{self.tag}.png"))
        plot_joint_angles(self.traj,
            os.path.join(self.output_dir, f"cup_task_joint_angles_{self.tag}.png"))
        plot_joint_velocities(self.traj,
            os.path.join(self.output_dir, f"cup_task_joint_velocities_{self.tag}.png"))

    def run_msk(self):
        print("\nExtracting muscle fiber lengths…")
        self.ms_lens = run_msk(self.traj, MODEL_PATH, MS_LABELS, normalise=True)
        lens_csv = os.path.join(self.output_dir, f"cup_task_fiber_lengths_{self.tag}.csv")
        self.ms_lens.to_csv(lens_csv, index=False)
        print(f"Fiber lengths: {lens_csv}")
        plot_muscle_lengths(self.ms_lens,
            os.path.join(self.output_dir, f"cup_task_fiber_lengths_{self.tag}.png"),
            os.path.join(self.output_dir, f"cup_task_fiber_length_change_{self.tag}.png"))
        return self.ms_lens

    # ── End-to-end ────────────────────────────────────────────────────────
    def run(self) -> dict:
        print(f"Parameters: T_MOVE={T_MOVE}s  D_MOVE={D_MOVE}m  "
              f"T_PERT={T_PERT}s  F_PERT={F_PERT}N  σ={SIGMA_PERT}s  m_eff={M_EFF}kg")

        print("\n── 1-D trajectory ──")
        print("\nLoading MoBL-ARMS 4.1…")
        self.init_model()

        print("Building perturbed trajectory (IK)…")
        self.build_trajectory()
        residuals = np.linalg.norm(self.actuals_3d - self.targets_3d, axis=1)
        print(f"IK residual  mean={residuals.mean()*1e3:.2f} mm  "
              f"max={residuals.max()*1e3:.2f} mm")

        out = self.save_trajectory()
        self.plot_all()
        self.run_msk()

        print("\nDone.")
        print("  Tip: vary perturbation parameters, e.g.")
        print("  python arm_cup_perturbation.py --t-pert 1.0 --f-pert 18 --sigma 0.03")
        return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Min-jerk cup trajectory with perturbation (purple-line reference)")
    ap.add_argument("--t-pert",  type=float, default=None,
                    help=f"Perturbation centre time [s]  (default {T_PERT})")
    ap.add_argument("--f-pert",  type=float, default=None,
                    help=f"Peak interaction force [N]    (default {F_PERT})")
    ap.add_argument("--sigma",   type=float, default=None,
                    help=f"Perturbation width σ [s]      (default {SIGMA_PERT})")
    ap.add_argument("--v-dip",   type=float, default=None,
                    help=f"Velocity dip fraction 0–1     (default {V_DIP_FRAC})")
    ap.add_argument("--m-eff",   type=float, default=None,
                    help=f"Effective endpoint mass [kg]  (default {M_EFF})")
    ap.add_argument("--d-move",  type=float, default=None,
                    help=f"Total displacement [m]        (default {D_MOVE})")
    ap.add_argument("--f-neg",   type=float, default=None,
                    help=f"Peak negative undershoot [N]  (default {F_NEG})")
    ap.add_argument("--dt-neg",  type=float, default=None,
                    help=f"Delay of undershoot after spike [s] (default {DT_NEG})")
    ap.add_argument("--sig-neg", type=float, default=None,
                    help=f"Width σ of negative lobe [s]  (default {SIG_NEG})")
    args = ap.parse_args()

    ArmCupPerturbation(
        t_pert=args.t_pert, f_pert=args.f_pert, sigma=args.sigma,
        v_dip=args.v_dip, m_eff=args.m_eff, d_move=args.d_move,
        f_neg=args.f_neg, dt_neg=args.dt_neg, sig_neg=args.sig_neg,
    ).run()
