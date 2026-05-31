"""
arm_cup_task.py

Drives the MoBL-ARMS 4.1 upper limb OpenSim model through a cup-and-ball task
trajectory and extracts muscle fiber lengths.

Two trajectory modes (set TRAJECTORY_MODE below):
  'sine'     – continuous sinusoidal oscillation (Lissajous rail path)
  'min_jerk' – minimum-jerk movements between named waypoints (Flash & Hogan 1985)

Usage:
  python arm_cup_task.py
"""

import os
import sys

# Ensure simbody-visualizer is on PATH (needed when VISUALIZE=True).
# The conda env ships the binary under libexec/simbody/.
_simbody_dir = os.path.join(sys.prefix, "libexec", "simbody")
if _simbody_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _simbody_dir + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import pandas as pd
import opensim as osim
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize as _sp_min

# ---------------------------------------------------------------------------
# Centralised constants (single source of truth)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MODEL_PATH, DEMO_OUTPUT_DIR,
    FS, ACTIVE_DOFS, NEUTRAL, MS_LABELS,
    HAND_BODY, HAND_LOCAL_PT,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
BASE_OUTPUT_DIR = DEMO_OUTPUT_DIR
OUTPUT_DIR      = os.path.join(DEMO_OUTPUT_DIR, "arm_cup_task")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Trajectory mode: 'sine' or 'min_jerk'
# ---------------------------------------------------------------------------
TRAJECTORY_MODE = 'sine'   # <-- change this to switch modes

# ---------------------------------------------------------------------------
# Visualisation flags
# ---------------------------------------------------------------------------
# VISUALIZE=True  – open Simbody 3-D window while the script runs
# OPEN_GUI=True   – open OpenSim GUI and load the .mot file after the run
VISUALIZE = False
OPEN_GUI  = False

# MS_LABELS, ACTIVE_DOFS, FS, NEUTRAL imported from config above.

# ===========================================================================
# MODE A – Sinusoidal (continuous Lissajous oscillation)
# ===========================================================================
DURATION = 8.0        # seconds
FREQ     = 0.5        # Hz – oscillation frequency on the rail

# Oscillation amplitudes (radians)
# shoulder_rot ±15° → left/right,  shoulder_elv ±10° → up/down
# 90° phase offset → circular Lissajous path on the rail.
# Set AMPLITUDES["shoulder_elv"] = 0 for a straight left-right rail.
AMPLITUDES = {
    "elv_angle":     0.0,
    "shoulder_elv":  np.radians(10),
    "shoulder_rot":  np.radians(15),
    "elbow_flexion": np.radians(5),
    "pro_sup":       np.radians(5),
    "deviation":     0.0,
    "flexion":       0.0,
}
PHASE = {
    "elv_angle":     0.0,
    "shoulder_elv":  0.0,
    "shoulder_rot":  np.pi / 2,
    "elbow_flexion": 0.0,
    "pro_sup":       np.pi / 2,
    "deviation":     0.0,
    "flexion":       0.0,
}


def build_trajectory_sine() -> pd.DataFrame:
    """Continuous sinusoidal oscillation around the neutral posture."""
    n = int(DURATION * FS)
    time = np.linspace(0, DURATION, n)
    rows = {"time": time}
    for dof in ACTIVE_DOFS:
        rows[dof] = (NEUTRAL[dof]
                     + AMPLITUDES[dof] * np.sin(2 * np.pi * FREQ * time + PHASE[dof]))
    return pd.DataFrame(rows)


# ===========================================================================
# MODE B – Minimum-jerk (waypoint-to-waypoint, Flash & Hogan 1985)
# ===========================================================================
# Each entry: (pose in degrees,  move_duration_s,  hold_duration_s)
#   move_duration = 0.0  → very first pose (no incoming movement)
# Waypoints are in degrees. Neutral base: elv=52.9, sev=87.4, srot=58.0,
# ef=65.6, ps=-0.9, dev=0.1, fl=-0.3.  Offsets preserved from original.
WAYPOINTS = [
    # pose (degrees)                                                                                              move  hold
    ({"elv_angle": 52.9, "shoulder_elv": 87.4, "shoulder_rot": 58.0, "elbow_flexion": 65.6, "pro_sup": -0.9, "deviation": 0.1, "flexion": -0.3}, 0.0, 0.5),  # neutral start
    ({"elv_angle": 52.9, "shoulder_elv": 92.4, "shoulder_rot": 43.0, "elbow_flexion": 63.6, "pro_sup": -5.9, "deviation": 0.1, "flexion": -0.3}, 0.8, 0.4),  # far left
    ({"elv_angle": 52.9, "shoulder_elv": 82.4, "shoulder_rot": 73.0, "elbow_flexion": 67.6, "pro_sup":  4.1, "deviation": 0.1, "flexion": -0.3}, 1.0, 0.4),  # far right
    ({"elv_angle": 52.9, "shoulder_elv": 97.4, "shoulder_rot": 63.0, "elbow_flexion": 60.6, "pro_sup": -0.9, "deviation": 0.1, "flexion": -0.3}, 0.8, 0.4),  # upper right
    ({"elv_angle": 52.9, "shoulder_elv": 77.4, "shoulder_rot": 53.0, "elbow_flexion": 70.6, "pro_sup": -0.9, "deviation": 0.1, "flexion": -0.3}, 0.9, 0.4),  # lower left
    ({"elv_angle": 52.9, "shoulder_elv": 87.4, "shoulder_rot": 58.0, "elbow_flexion": 65.6, "pro_sup": -0.9, "deviation": 0.1, "flexion": -0.3}, 0.8, 0.5),  # return neutral
    ({"elv_angle": 52.9, "shoulder_elv": 95.4, "shoulder_rot": 48.0, "elbow_flexion": 62.6, "pro_sup": -3.9, "deviation": 0.1, "flexion": -0.3}, 0.7, 0.3),  # second left
    ({"elv_angle": 52.9, "shoulder_elv": 79.4, "shoulder_rot": 70.0, "elbow_flexion": 68.6, "pro_sup":  2.1, "deviation": 0.1, "flexion": -0.3}, 0.9, 0.4),  # second right
    ({"elv_angle": 52.9, "shoulder_elv": 87.4, "shoulder_rot": 58.0, "elbow_flexion": 65.6, "pro_sup": -0.9, "deviation": 0.1, "flexion": -0.3}, 0.8, 0.5),  # return neutral
]


def _min_jerk_basis(tau: np.ndarray) -> np.ndarray:
    """Minimum-jerk position basis.  tau in [0, 1] → smooth S-curve."""
    tau = np.clip(tau, 0.0, 1.0)
    return 10*tau**3 - 15*tau**4 + 6*tau**5


def build_trajectory_min_jerk() -> pd.DataFrame:
    """
    Minimum-jerk trajectory through WAYPOINTS (Flash & Hogan 1985).
    Angles in WAYPOINTS are degrees; output is in radians.
    """
    segments = []  # list of (time_array, {dof: angle_array_rad})

    for i in range(len(WAYPOINTS) - 1):
        pose_start, _, hold_dur = WAYPOINTS[i]
        pose_end,   move_dur, _ = WAYPOINTS[i + 1]

        if hold_dur > 0:
            n = max(1, int(hold_dur * FS))
            t = np.linspace(0, hold_dur, n, endpoint=False)
            angles = {dof: np.full(n, np.radians(pose_start[dof])) for dof in ACTIVE_DOFS}
            segments.append((t, angles))

        if move_dur > 0:
            n = max(2, int(move_dur * FS))
            t = np.linspace(0, move_dur, n, endpoint=False)
            basis = _min_jerk_basis(t / move_dur)
            angles = {}
            for dof in ACTIVE_DOFS:
                a0 = np.radians(pose_start[dof])
                af = np.radians(pose_end[dof])
                angles[dof] = a0 + (af - a0) * basis
            segments.append((t, angles))

    pose_last, _, hold_last = WAYPOINTS[-1]
    n = max(1, int(hold_last * FS))
    t = np.linspace(0, hold_last, n, endpoint=False)
    segments.append((t, {dof: np.full(n, np.radians(pose_last[dof])) for dof in ACTIVE_DOFS}))

    time_out = []
    dof_out  = {dof: [] for dof in ACTIVE_DOFS}
    t_offset = 0.0
    for t_seg, angles_seg in segments:
        time_out.append(t_seg + t_offset)
        for dof in ACTIVE_DOFS:
            dof_out[dof].append(angles_seg[dof])
        t_offset += t_seg[-1] + 1.0 / FS

    rows = {"time": np.concatenate(time_out)}
    for dof in ACTIVE_DOFS:
        rows[dof] = np.concatenate(dof_out[dof])
    return pd.DataFrame(rows)


# ===========================================================================
# Dispatcher
# ===========================================================================
def build_trajectory(mode: str = TRAJECTORY_MODE) -> pd.DataFrame:
    """Return trajectory DataFrame for the chosen mode ('sine' or 'min_jerk')."""
    if mode == 'sine':
        return build_trajectory_sine()
    elif mode == 'min_jerk':
        return build_trajectory_min_jerk()
    else:
        raise ValueError(f"Unknown TRAJECTORY_MODE '{mode}'. Choose 'sine' or 'min_jerk'.")


# ---------------------------------------------------------------------------
# Write OpenSim .mot file
# ---------------------------------------------------------------------------
def write_mot(traj: pd.DataFrame, path: str):
    header = (
        "cup_task_trajectory\n"
        "version=1\n"
        f"nRows={len(traj)}\n"
        f"nColumns={len(traj.columns)}\n"
        "inDegrees=no\n"
        "endheader\n"
    )
    with open(path, "w") as f:
        f.write(header)
    traj.to_csv(path, sep="\t", index=False, mode="a")
    print(f"Wrote motion file: {path}")


# ---------------------------------------------------------------------------
# Shared Static Optimisation helpers
# (imported by arm_inverse_dynamics.py and compute_stiffness.py)
# ---------------------------------------------------------------------------

def compute_moment_arms(model, state, ms_labels: list,
                        dofs: list) -> np.ndarray:
    """
    Moment arm matrix  R ∈ ℝ^{n_dofs × n_muscles}.
    R[i, j] = moment arm of muscle j about DOF i (metres/rad).
    State must be at least Position-realised.
    """
    n_dofs = len(dofs)
    n_ms   = len(ms_labels)
    R = np.zeros((n_dofs, n_ms))
    for i, dof in enumerate(dofs):
        coord = model.getCoordinateSet().get(dof)
        for j, ms_name in enumerate(ms_labels):
            R[i, j] = model.getMuscles().get(ms_name).computeMomentArm(state, coord)
    return R


def so_frame(tau:         np.ndarray,
             R:           np.ndarray,
             F0:          np.ndarray,
             fl:          np.ndarray,
             alpha_floor: float = 0.0,
             x0:          np.ndarray = None,
             cost_exp:    int   = 3) -> np.ndarray:
    """
    Per-frame static optimisation (Crowninshield–Brand by default).

    minimise  Σ aᵢ^p           (p = cost_exp; default 3)
    subject to  R · (a ⊙ F0 ⊙ fl) = tau,   alpha_floor ≤ aᵢ ≤ 1

    p = 3 spreads activation across synergists and avoids the bang-bang
    saturation typical of p = 2; this is the classical recruitment cost
    used in OpenSim's Static Optimisation tool.

    Parameters
    ----------
    tau         : (n_dofs,)       required joint torques  [N·m]
    R           : (n_dofs, n_ms)  moment arm matrix
    F0          : (n_ms,)         max isometric forces    [N]
    fl          : (n_ms,)         force–length multiplier (≥ 0)
    alpha_floor : float           minimum activation (0 = pure SO;
                                  > 0 = CMC null-space co-contraction floor)
    x0          : (n_ms,) or None warm-start activations (default: zeros)
    cost_exp    : int             cost exponent (2 = min-effort, 3 = Crowninshield–Brand)

    Returns
    -------
    a : (n_ms,) activations ∈ [alpha_floor, 1]
    """
    n_ms    = len(F0)
    A       = R * (F0 * fl)[np.newaxis, :]         # (n_dofs, n_ms)
    if x0 is None:
        x0 = np.zeros(n_ms)
    p = int(cost_exp)

    res = _sp_min(
        lambda a: float(np.sum(a ** p)),
        x0      = np.clip(x0, alpha_floor, 1.0),
        jac     = lambda a: p * (a ** (p - 1)),
        method  = "SLSQP",
        bounds  = [(float(alpha_floor), 1.0)] * n_ms,
        constraints = {"type": "eq",
                       "fun":  lambda a: A @ a - tau,
                       "jac":  lambda a: A},
        options = {"ftol": 1e-8, "maxiter": 400, "disp": False},
    )
    return np.clip(res.x, alpha_floor, 1.0)


# ---------------------------------------------------------------------------
# Drive the MSK model and extract normalised fiber lengths
# ---------------------------------------------------------------------------
def run_msk(traj: pd.DataFrame, model_path: str, ms_labels: list,
            normalise: bool = True) -> pd.DataFrame:
    print(f"Loading model: {model_path}")
    model = osim.Model(model_path)
    if VISUALIZE:
        model.setUseVisualizer(True)
    state = model.initSystem()

    # Optimal fiber lengths (model constants — physiologically correct reference)
    l_opt = {ms: model.getMuscles().get(ms).getOptimalFiberLength()
             for ms in ms_labels}

    # Diagnostic: neutral-pose passive fiber length vs l_opt
    print("Computing neutral-pose fiber lengths for diagnostic…")
    for dof in ACTIVE_DOFS:
        model.updCoordinateSet().get(dof).setValue(state, NEUTRAL[dof], False)
    model.assemble(state)
    model.equilibrateMuscles(state)
    print(f"  {'Muscle':<12}  {'l_neutral (m)':>14}  {'l_opt (m)':>10}  {'ratio':>7}")
    for ms in ms_labels:
        l_n = model.getMuscles().get(ms).getFiberLength(state)
        lo  = l_opt[ms]
        flag = "  ← FAR FROM OPTIMAL" if abs(l_n / lo - 1.0) > 0.35 else ""
        print(f"  {ms:<12}  {l_n:>14.4f}  {lo:>10.4f}  {l_n/lo:>7.3f}{flag}")

    # Iterate timesteps
    records = []
    for i in tqdm(range(len(traj)), desc="Extracting muscle fiber lengths"):
        row = traj.iloc[i]
        for dof in ACTIVE_DOFS:
            model.updCoordinateSet().get(dof).setValue(state, row[dof])
        model.realizePosition(state)
        model.equilibrateMuscles(state)

        if VISUALIZE:
            model.getVisualizer().show(state)

        rec = {"time": row["time"]}
        for ms in ms_labels:
            rec[ms] = model.getMuscles().get(ms).getFiberLength(state)
        records.append(rec)

    ms_lens = pd.DataFrame(records)

    if normalise:
        for ms in ms_labels:
            ms_lens[ms] = ms_lens[ms] / l_opt[ms]

    return ms_lens


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_trajectory(traj: pd.DataFrame, save_path: str):
    fig, axes = plt.subplots(len(ACTIVE_DOFS), 1, figsize=(10, 12), sharex=True)
    for ax, dof in zip(axes, ACTIVE_DOFS):
        ax.plot(traj["time"], np.degrees(traj[dof]))
        ax.set_ylabel(f"{dof}\n(deg)", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Cup-task arm trajectory ({TRAJECTORY_MODE})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved trajectory plot: {save_path}")


def plot_fiber_lengths(ms_lens: pd.DataFrame, save_path: str):
    ms_cols = [c for c in ms_lens.columns if c != "time"]
    n = len(ms_cols)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 2.5), sharex=True)
    axes = axes.flatten()
    for ax, ms in zip(axes, ms_cols):
        ax.plot(ms_lens["time"], ms_lens[ms])
        ax.set_title(ms, fontsize=9)
        ax.set_ylabel("L / L\u2080", fontsize=8)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.grid(True, alpha=0.3)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[max(0, n - n_cols):n]:
        ax.set_xlabel("Time (s)", fontsize=8)
    fig.suptitle(f"Normalised muscle fiber lengths \u2013 cup task ({TRAJECTORY_MODE})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved fiber-length plot: {save_path}")


# ===========================================================================
# Class wrapper — ArmCupTask
# ===========================================================================
class ArmCupTask:
    """Object-oriented driver for the cup-task fiber-length pipeline.

    Wraps the module-level functions (build_trajectory, run_msk, plot_*) into
    a single class with state. Module-level functions remain available for
    backward compatibility with downstream scripts that import them.
    """

    def __init__(
        self,
        trajectory_mode: str = TRAJECTORY_MODE,
        model_path: str = MODEL_PATH,
        ms_labels: list = None,
        output_dir: str = OUTPUT_DIR,
        open_gui: bool = OPEN_GUI,
    ):
        self.trajectory_mode = trajectory_mode
        self.model_path      = model_path
        self.ms_labels       = list(ms_labels) if ms_labels is not None else list(MS_LABELS)
        self.output_dir      = output_dir
        self.open_gui        = open_gui
        os.makedirs(self.output_dir, exist_ok=True)

        # Populated by run()
        self.traj    = None
        self.ms_lens = None

    # ── Trajectory ────────────────────────────────────────────────────────
    def build_trajectory(self) -> pd.DataFrame:
        self.traj = build_trajectory(self.trajectory_mode)
        return self.traj

    # ── MSK forward pass ──────────────────────────────────────────────────
    def run_msk(self, normalise: bool = True) -> pd.DataFrame:
        if self.traj is None:
            self.build_trajectory()
        self.ms_lens = run_msk(self.traj, self.model_path, self.ms_labels,
                               normalise=normalise)
        return self.ms_lens

    # ── Outputs ───────────────────────────────────────────────────────────
    def save_outputs(self, tag: str = None) -> dict:
        if tag is None:
            tag = self.trajectory_mode
        if self.traj is None or self.ms_lens is None:
            raise RuntimeError("Call build_trajectory()+run_msk() (or run()) first.")

        out = {}
        out["traj_csv"] = os.path.join(self.output_dir, f"cup_task_trajectory_{tag}.csv")
        out["mot"]      = os.path.join(self.output_dir, f"cup_task_trajectory_{tag}.mot")
        out["traj_png"] = os.path.join(self.output_dir, f"cup_task_trajectory_{tag}.png")
        out["lens_csv"] = os.path.join(self.output_dir, f"cup_task_fiber_lengths_{tag}.csv")
        out["lens_png"] = os.path.join(self.output_dir, f"cup_task_fiber_lengths_{tag}.png")

        self.traj.to_csv(out["traj_csv"], index=False)
        print(f"Trajectory saved: {out['traj_csv']}  ({len(self.traj)} frames @ {FS} Hz)")
        write_mot(self.traj, out["mot"])
        plot_trajectory(self.traj, out["traj_png"])
        self.ms_lens.to_csv(out["lens_csv"], index=False)
        print(f"Fiber lengths saved: {out['lens_csv']}")
        plot_fiber_lengths(self.ms_lens, out["lens_png"])
        return out

    # ── End-to-end ────────────────────────────────────────────────────────
    def run(self) -> dict:
        print(f"Trajectory mode: {self.trajectory_mode}")
        self.build_trajectory()
        self.run_msk(normalise=True)
        out = self.save_outputs(tag=self.trajectory_mode)

        print("\nDone. Next step: feed ms_lens into NeuroMotion MNPool pipeline.")
        print("  See NeuroMotion/scripts/mov2emg.py for downstream EMG generation.")

        if self.open_gui:
            import subprocess
            gui_app = "/Applications/OpenSim 4.5/OpenSim.app"
            print("\nOpening OpenSim GUI…")
            subprocess.Popen([
                "open", "-a", gui_app,
                "--args",
                "-ModelFile",  self.model_path,
                "-MotionFile", out["mot"],
            ])
        return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ArmCupTask().run()
