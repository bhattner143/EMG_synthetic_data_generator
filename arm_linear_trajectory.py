"""
arm_linear_trajectory.py

Elevation-plane LINEAR trajectory for MoBL-ARMS 4.1.

The arm is elevated at the fixed shoulder_elv neutral angle, and
shoulder_rot is locked.  elv_angle and elbow_flexion are solved jointly
via numerical IK so that the hand endpoint traces a straight
left-right line (azimuthal sweep in the horizontal plane).

  Locked : shoulder_elv, shoulder_rot, pro_sup, deviation, flexion  (at neutral)
  Active : elv_angle + elbow_flexion  (IK-solved each frame)

Motion  : sinusoidal, LINE_FREQ Hz, ±LINE_AMP metres along the in-plane line

Outputs (demo_output/):
  cup_task_trajectory_linear.csv / .mot
  cup_task_fiber_lengths_linear.csv
  cup_task_trajectory_linear.png          – active DOF time series
  cup_task_endpoint_path_linear.png       – 3-D hand path vs target line

Usage:
  python arm_linear_trajectory.py
"""

import os
import sys

_simbody_dir = os.path.join(sys.prefix, "libexec", "simbody")
if _simbody_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _simbody_dir + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import pandas as pd
import opensim as osim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3-D projection
from tqdm import tqdm
from scipy.optimize import minimize as sp_minimize
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_cup_task import (
    MODEL_PATH, DEMO_OUTPUT_DIR, MS_LABELS, ACTIVE_DOFS, NEUTRAL, FS,
    write_mot, run_msk,
)
import os as _os
SCRIPT_NAME = _os.path.splitext(_os.path.basename(__file__))[0]
OUTPUT_DIR  = _os.path.join(DEMO_OUTPUT_DIR, SCRIPT_NAME)
_os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
TAG       = "linear"
DURATION  = 8.0    # seconds
LINE_FREQ = 0.5    # Hz  – oscillation frequency along the line
LINE_AMP  = 0.15   # m   – half-amplitude (total travel = 2 × AMP)

# Hand/cup endpoint: body name and local station in that body's frame (metres).
# In MoBL-ARMS 4.1 the wrist-centre body is "hand"; change HAND_BODY if needed.
HAND_BODY     = "hand"
HAND_LOCAL_PT = (0.0, 0.0, 0.0)   # wrist joint centre; shift distally to adjust

# IK joint limits (radians)
ELV_ANGLE_BOUNDS  = (np.radians(-30), np.radians(120))
ELBOW_FLEX_BOUNDS = (np.radians(25),  np.radians(160))

# IK is solved on a coarse grid then spline-interpolated to full FS.
# IK_FS=10 means 80 IK solves for 8 s → fast; spline fills in the rest.
IK_FS = 10   # Hz – coarse IK sample rate (must divide FS evenly or be ≤ FS)

# Set True to open Simbody 3-D window while building the trajectory
VISUALIZE = False

# ---------------------------------------------------------------------------
# Locked DOF values (everything except elv_angle and elbow_flexion)
# ---------------------------------------------------------------------------
LOCKED = {
    "shoulder_elv": NEUTRAL["shoulder_elv"],
    "shoulder_rot": NEUTRAL["shoulder_rot"],
    "pro_sup":      NEUTRAL["pro_sup"],
    "deviation":    NEUTRAL["deviation"],
    "flexion":      NEUTRAL["flexion"],
}

# ===========================================================================
# OpenSim helpers
# ===========================================================================

def _init_model() -> tuple:
    """Load model, lock non-IK DOFs, return (model, state)."""
    m = osim.Model(MODEL_PATH)
    m.setUseVisualizer(VISUALIZE)
    s = m.initSystem()
    for dof, val in LOCKED.items():
        m.updCoordinateSet().get(dof).setValue(s, val, False)
    m.assemble(s)
    return m, s


def _hand_pos(model: osim.Model, state, q: np.ndarray) -> np.ndarray:
    """
    FK: set shoulder_elv=q[0], elbow_flexion=q[1], realise position,
    return hand station position in ground frame as shape-(3,) array.

    NOTE: setValue(..., False) skips per-set constraint enforcement.
    realizePosition() propagates kinematics without a full assembly solve,
    which is ~100× faster inside the IK optimiser hot-path.
    The locked DOFs were assembled once in _init_model() and do not change.
    """
    model.updCoordinateSet().get("elv_angle").setValue(state, q[0], False)
    model.updCoordinateSet().get("elbow_flexion").setValue(state, q[1], False)
    model.realizePosition(state)
    hand    = model.getBodySet().get(HAND_BODY)
    p_local = osim.Vec3(*HAND_LOCAL_PT)
    p_gnd   = hand.findStationLocationInGround(state, p_local)
    return np.array([p_gnd[0], p_gnd[1], p_gnd[2]])


def _ik_cost(q: np.ndarray, model, state, target: np.ndarray) -> float:
    return float(np.sum((_hand_pos(model, state, q) - target) ** 2))


def solve_ik(
    model: osim.Model,
    state,
    target: np.ndarray,
    q0: np.ndarray,
) -> np.ndarray:
    """
    Minimise ||FK(elv_angle, elbow_flexion) - target||² via L-BFGS-B.
    Warm-started from q0.  Returns (elv_angle, elbow_flexion) in radians.
    """
    res = sp_minimize(
        _ik_cost,
        x0=q0,
        args=(model, state, target),
        method="L-BFGS-B",
        bounds=[ELV_ANGLE_BOUNDS, ELBOW_FLEX_BOUNDS],
        options={"ftol": 1e-14, "gtol": 1e-8, "maxiter": 300},
    )
    return res.x


# ===========================================================================
# Line definition
# ===========================================================================

def compute_line(model: osim.Model, state) -> tuple:
    """
    Determine the straight-line centre and direction in the ground frame.

    Strategy: sample the hand endpoint at elv_angle ±10° around neutral
    with shoulder_elv and elbow_flexion fixed at neutral.  Varying elv_angle
    sweeps the arm's plane of elevation left-right (azimuthal sweep), so
    the resulting direction vector is predominantly horizontal.

    Returns:
        center    (3,)  – endpoint at neutral posture
        direction (3,)  – unit vector along the left-right sweep direction
    """
    q_neutral = np.array([NEUTRAL["elv_angle"],  NEUTRAL["elbow_flexion"]])
    q_low     = np.array([NEUTRAL["elv_angle"] - np.radians(10.0),
                          NEUTRAL["elbow_flexion"]])
    q_high    = np.array([NEUTRAL["elv_angle"] + np.radians(10.0),
                          NEUTRAL["elbow_flexion"]])

    center    = _hand_pos(model, state, q_neutral)
    p_low     = _hand_pos(model, state, q_low)
    p_high    = _hand_pos(model, state, q_high)

    diff      = p_high - p_low
    direction = diff / np.linalg.norm(diff)

    print(f"  Hand centre (neutral) : {center}")
    print(f"  Line direction        : {direction}  (ground frame)")
    print(f"  Sweep span at ±10°   : {np.linalg.norm(diff)*100:.1f} cm")
    return center, direction


# ===========================================================================
# Trajectory builder
# ===========================================================================

def build_linear_trajectory(model: osim.Model, state) -> tuple:
    """
    Solve IK on a coarse IK_FS-Hz grid then cubic-spline interpolate to FS.

    Hand endpoint target:
        target(t) = center + LINE_AMP * sin(2π LINE_FREQ t) * direction

    Returns:
        traj      (DataFrame) – columns [time] + ACTIVE_DOFS  (@ FS Hz)
        center    (3,)
        direction (3,)
        targets   (N, 3)      – desired endpoint positions    (@ FS Hz)
        actuals   (N, 3)      – achieved FK positions after IK (@ FS Hz)
    """
    center, direction = compute_line(model, state)

    # --- coarse IK grid ---------------------------------------------------
    n_coarse   = int(DURATION * IK_FS) + 1
    t_coarse   = np.linspace(0, DURATION, n_coarse)
    tgt_coarse = (center[np.newaxis, :]
                  + LINE_AMP
                  * np.sin(2 * np.pi * LINE_FREQ * t_coarse)[:, np.newaxis]
                  * direction[np.newaxis, :])

    q = np.array([NEUTRAL["elv_angle"], NEUTRAL["elbow_flexion"]])
    q_coarse = np.zeros((n_coarse, 2))

    for i, tgt in enumerate(
        tqdm(tgt_coarse, total=n_coarse, desc="IK – coarse grid")
    ):
        q = solve_ik(model, state, tgt, q0=q)
        q_coarse[i] = q

    # --- spline interpolate to full FS ------------------------------------
    cs_elv_angle = CubicSpline(t_coarse, q_coarse[:, 0])
    cs_flex      = CubicSpline(t_coarse, q_coarse[:, 1])

    n      = int(DURATION * FS)
    time   = np.linspace(0, DURATION, n)
    targets = (center[np.newaxis, :]
               + LINE_AMP
               * np.sin(2 * np.pi * LINE_FREQ * time)[:, np.newaxis]
               * direction[np.newaxis, :])

    elv_angle_full = cs_elv_angle(time)
    flex_full      = cs_flex(time)

    # clip to joint limits after interpolation
    elv_angle_full = np.clip(elv_angle_full, *ELV_ANGLE_BOUNDS)
    flex_full      = np.clip(flex_full,      *ELBOW_FLEX_BOUNDS)

    # --- verify FK on full grid (fast: no optimizer, just realizePosition) -
    actuals = np.zeros((n, 3))
    for i, (a, f) in enumerate(zip(elv_angle_full, flex_full)):
        actuals[i] = _hand_pos(model, state, np.array([a, f]))

    records = []
    for t_i, a, f in zip(time, elv_angle_full, flex_full):
        row = {"time": t_i, "elv_angle": a, "elbow_flexion": f}
        row.update(LOCKED)
        records.append(row)

    traj = pd.DataFrame(records)[["time"] + ACTIVE_DOFS]
    return traj, center, direction, targets, actuals


# ===========================================================================
# Plots
# ===========================================================================

def plot_active_dofs(traj: pd.DataFrame, save_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for ax, dof in zip(axes, ["elv_angle", "elbow_flexion"]):
        ax.plot(traj["time"], np.degrees(traj[dof]))
        ax.set_ylabel(f"{dof} (deg)", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Linear trajectory – IK-solved active DOFs")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_endpoint_path(
    center: np.ndarray,
    direction: np.ndarray,
    targets: np.ndarray,
    actuals: np.ndarray,
    save_path: str,
):
    """3-D plot: target line vs achieved FK hand path."""
    fig = plt.figure(figsize=(9, 6))
    ax  = fig.add_subplot(111, projection="3d")

    ax.plot(targets[:, 0], targets[:, 1], targets[:, 2],
            "r--", linewidth=1.5, label="Target line")
    ax.plot(actuals[:, 0], actuals[:, 1], actuals[:, 2],
            "b-",  linewidth=1.5, label="IK path (FK verified)")
    ax.scatter(*center, color="k", s=60, zorder=5, label="Neutral centre")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Hand endpoint: target line vs IK-achieved path")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_fiber_lengths(ms_lens: pd.DataFrame, save_path: str,
                       title: str = "Normalised muscle lengths – linear trajectory"):
    ms_cols = [c for c in ms_lens.columns if c != "time"]
    n_cols  = 3
    n_rows  = (len(ms_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(14, n_rows * 2.5), sharex=True)
    axes = axes.flatten()
    for ax, ms in zip(axes, ms_cols):
        vals = ms_lens[ms].values
        ax.plot(ms_lens["time"], vals)
        # highlight haywire frames: deviation > 3× IQR from median
        med = np.median(vals)
        iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
        threshold = 3 * max(iqr, 0.01)
        bad = np.abs(vals - med) > threshold
        if bad.any():
            ax.scatter(ms_lens["time"][bad], vals[bad],
                       color="red", s=12, zorder=5, label="spike")
            ax.legend(fontsize=7, loc="upper right")
        ax.set_title(ms, fontsize=9)
        ax.set_ylabel("L / L₀", fontsize=8)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(ms_cols):]:
        ax.set_visible(False)
    for ax in axes[max(0, len(ms_cols) - n_cols):len(ms_cols)]:
        ax.set_xlabel("Time (s)", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ===========================================================================
# Fast musculotendon-length check  (purely kinematic, no equilibration)
# ===========================================================================

def fast_muscle_lengths(traj: pd.DataFrame,
                        labels: list | None = None) -> pd.DataFrame:
    """
    Compute musculotendon lengths for every frame using realizePosition only
    (no equilibrateMuscles).  Returns a DataFrame normalised by neutral length.

    labels: list of muscle names to include.  If None, all muscles in the
            model are used.

    This runs at ~FK speed (~800 frames/s) and is useful for spotting
    frames where geometry goes haywire before committing to the slow
    equilibration-based run.
    """
    model = osim.Model(MODEL_PATH)
    state = model.initSystem()

    # resolve muscle list
    if labels is None:
        muscles = model.getMuscles()
        labels = [muscles.get(i).getName() for i in range(muscles.getSize())]

    # neutral reference lengths
    for dof in ACTIVE_DOFS:
        model.updCoordinateSet().get(dof).setValue(state, NEUTRAL[dof], False)
    model.assemble(state)
    model.realizePosition(state)
    ref = {ms: model.getMuscles().get(ms).getLength(state) for ms in labels}

    records = []
    for i in tqdm(range(len(traj)), desc="Fast muscle lengths", ncols=80):
        row = traj.iloc[i]
        for dof in ACTIVE_DOFS:
            model.updCoordinateSet().get(dof).setValue(state, row[dof], False)
        model.realizePosition(state)
        rec = {"time": row["time"]}
        for ms in labels:
            rec[ms] = model.getMuscles().get(ms).getLength(state) / ref[ms]
        records.append(rec)

    return pd.DataFrame(records)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fibers", action="store_true",
                    help="Skip the slow muscle fiber-length extraction")
    ap.add_argument("--fast-fibers", action="store_true",
                    help="Quick kinematic muscle-length check (no equilibration, ~1s)")
    ap.add_argument("--all-muscles", action="store_true",
                    help="Plot every muscle in the model (use with --fast-fibers)")
    args = ap.parse_args()

    print("Loading MoBL-ARMS 4.1…")
    model, state = _init_model()

    # 1. Build IK trajectory
    print("Building linear trajectory (IK)…")
    traj, center, direction, targets, actuals = build_linear_trajectory(model, state)

    # IK residual report
    residuals = np.linalg.norm(actuals - targets, axis=1)
    print(f"IK residual  mean={residuals.mean()*1e3:.2f} mm  "
          f"max={residuals.max()*1e3:.2f} mm")

    # 2. Save CSV
    traj_csv = os.path.join(OUTPUT_DIR, f"cup_task_trajectory_{TAG}.csv")
    traj.to_csv(traj_csv, index=False)
    print(f"Trajectory : {traj_csv}  ({len(traj)} frames @ {FS} Hz)")

    # 3. Write .mot
    mot_path = os.path.join(OUTPUT_DIR, f"cup_task_trajectory_{TAG}.mot")
    write_mot(traj, mot_path)

    # 4. Plot active DOFs
    plot_active_dofs(
        traj,
        os.path.join(OUTPUT_DIR, f"cup_task_trajectory_{TAG}.png"),
    )

    # 5. Plot endpoint path (target vs achieved)
    plot_endpoint_path(
        center, direction, targets, actuals,
        os.path.join(OUTPUT_DIR, f"cup_task_endpoint_path_{TAG}.png"),
    )

    if args.fast_fibers:
        # determine which muscle set to use
        muscle_labels = None if args.all_muscles else MS_LABELS
        label_tag = "all" if args.all_muscles else "emg"

        # Quick kinematic check — no equilibration, runs in ~1 s
        print(f"Running fast kinematic muscle-length check "
              f"({'all muscles' if args.all_muscles else f'{len(MS_LABELS)} EMG muscles'})…")
        ms_lens = fast_muscle_lengths(traj, labels=muscle_labels)
        lens_csv = os.path.join(OUTPUT_DIR, f"cup_task_fiber_lengths_{TAG}.csv")
        ms_lens.to_csv(lens_csv, index=False)
        print(f"Saved: {lens_csv}")

        # Print any haywire frames
        ms_cols = [c for c in ms_lens.columns if c != "time"]
        print(f"\nSpike report ({len(ms_cols)} muscles, threshold = 3×IQR):")
        n_spiking = 0
        for ms in ms_cols:
            vals = ms_lens[ms].values
            med = np.median(vals)
            iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
            threshold = 3 * max(iqr, 0.01)
            bad_idx = np.where(np.abs(vals - med) > threshold)[0]
            if len(bad_idx):
                n_spiking += 1
                print(f"  SPIKE  {ms:14s}: frames {bad_idx.tolist()} "
                      f"(t={ms_lens['time'].values[bad_idx].round(2).tolist()})  "
                      f"values={vals[bad_idx].round(3).tolist()}")
        if n_spiking == 0:
            print("  No spikes found — all muscles vary smoothly.")

        plot_fiber_lengths(
            ms_lens,
            os.path.join(OUTPUT_DIR, f"cup_task_fiber_lengths_{TAG}_{label_tag}.png"),
            title=f"Musculotendon lengths (kinematic, normalised) – {label_tag} – linear trajectory",
        )
        print(f"Saved plot: cup_task_fiber_lengths_{TAG}_{label_tag}.png")
    elif not args.no_fibers:
        # 6. MSK pipeline → normalised fiber lengths
        ms_lens = run_msk(traj, MODEL_PATH, MS_LABELS, normalise=True)
        lens_csv = os.path.join(OUTPUT_DIR, f"cup_task_fiber_lengths_{TAG}.csv")
        ms_lens.to_csv(lens_csv, index=False)
        print(f"Fiber lengths: {lens_csv}")

        # 7. Plot fiber lengths
        plot_fiber_lengths(
            ms_lens,
            os.path.join(OUTPUT_DIR, f"cup_task_fiber_lengths_{TAG}.png"),
        )
    else:
        print("Skipped fiber-length extraction (--no-fibers).")

    print("\nDone.")
    print("  To add perturbations + endpoint stiffness:")
    print("  python compute_stiffness.py --mode linear --stiffness cmc --perturb")
