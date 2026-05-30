"""
compute_stiffness.py

Endpoint stiffness computation for the MoBL-ARMS cup-task pipeline.
Extends arm_cup_task.py with two stiffness modes:

  static_opt  – Inverse Dynamics + Static Optimization at each frame.
                Muscle activations are the minimum-effort solution that
                produces the required joint torques.  K_e(t) follows
                directly from those activations and the muscle geometry.

  cmc         – Same as static_opt but adds a co-contraction activation
                floor (COCONTRACTION_ALPHA) during perturbation windows.
                This approximates the defensive muscle stiffening that a
                participant would produce when the ball is disturbed
                (analogous to the CMC null-space co-contraction signal that
                drives c_exo in the tele-impedance framework).

Outputs (saved to demo_output/):
  cup_task_trajectory_<tag>.csv / .mot
  cup_task_fiber_lengths_<tag>.csv
  cup_task_stiffness_<tag>.csv        ← new
  cup_task_stiffness_<tag>.png        ← new

Usage:
  # fiber lengths only (delegates to arm_cup_task logic)
  python compute_stiffness.py --mode sine

  # Static Optimization stiffness on a perturbed min-jerk trial
  python compute_stiffness.py --mode min_jerk --stiffness static_opt --perturb

  # CMC-approximation mode (perturbations auto-enabled)
  python compute_stiffness.py --mode sine --stiffness cmc
"""

import os
import sys
import argparse

# ── Simbody visualiser path (same fix as arm_cup_task.py) ──────────────────
_simbody_dir = os.path.join(sys.prefix, "libexec", "simbody")
if _simbody_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _simbody_dir + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import pandas as pd
import opensim as osim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interp1d as sp_interp1d

# ── Import shared constants and trajectory builders from arm_cup_task ───────
# (keeps everything DRY; arm_cup_task.py must live in the same folder)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_cup_task import (
    MODEL_PATH, DEMO_OUTPUT_DIR, MS_LABELS, ACTIVE_DOFS, NEUTRAL, FS,
    build_trajectory, write_mot, run_msk,
    plot_trajectory, plot_fiber_lengths,
    compute_moment_arms, so_frame,
)
from config import (
    BASELINE_ALPHA, COCONTRACTION_ALPHA,
    HAND_BODY, FL_GAMMA, BETA_D, V_MAX_FACTOR,
    FS, T_PERT, SIGMA_PERT,
)

# Activation dynamics low-pass time constant (s).
# Combined electromechanical delay (~30 ms) + activation/deactivation
# dynamics (~30–50 ms). Renders SO output physiologically realistic.
TAU_ACT = 0.040

# Width (s) of the smooth Gaussian co-contraction ramp around T_PERT.
# Wider than SIGMA_PERT (the force spike width) so the floor builds up
# before the perturbation peak and decays after, mimicking pre-activation.
SIGMA_FLOOR = 0.080
import os as _os
SCRIPT_NAME = _os.path.splitext(_os.path.basename(__file__))[0]
OUTPUT_DIR  = _os.path.join(DEMO_OUTPUT_DIR, SCRIPT_NAME)
_os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STIFFNESS_MODE = "none"          # overridden by --stiffness

# Perturbation parameters (script-specific)
PERTURB_TIMES    = [2.5, 5.0]    # seconds: bump injection times
PERTURB_DURATION = 0.15          # seconds: half-sine bump width
PERTURB_MAG_DEG  = 8.0           # degrees: shoulder_rot displacement amplitude
# Hill + activation constants (BASELINE_ALPHA, COCONTRACTION_ALPHA, HAND_BODY,
# FL_GAMMA, BETA_D, V_MAX_FACTOR) imported from config.py above.


# ===========================================================================
# Perturbation injection
# ===========================================================================

def add_perturbations(
    traj: pd.DataFrame,
    perturb_times: list = None,
    duration: float = PERTURB_DURATION,
    mag_deg: float = PERTURB_MAG_DEG,
) -> tuple:
    """
    Inject half-sine displacement bumps into shoulder_rot at each time in
    perturb_times.  Returns (perturbed_traj, bool_mask) where mask is True
    during each bump window.
    """
    if perturb_times is None:
        perturb_times = PERTURB_TIMES

    traj_p = traj.copy()
    t = traj_p["time"].to_numpy()
    mask = np.zeros(len(t), dtype=bool)

    for pt in perturb_times:
        win = (t >= pt) & (t < pt + duration)
        if not win.any():
            continue
        t_win = t[win] - pt
        bump = np.radians(mag_deg) * np.sin(np.pi * t_win / duration)
        traj_p.loc[win, "shoulder_rot"] = (
            traj_p.loc[win, "shoulder_rot"].to_numpy() + bump
        )
        mask[win] = True

    return traj_p, mask


# ===========================================================================
# Inverse Dynamics (OpenSim tool wrapper)
# ===========================================================================

def _read_sto(path: str) -> pd.DataFrame:
    """Read an OpenSim .sto/.mot file, skipping header up to 'endheader'."""
    with open(path) as fh:
        lines = fh.readlines()
    skip = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            skip = i + 1
            break
    return pd.read_csv(path, sep="\t", skiprows=skip)


def _extract_active_torques(id_df: pd.DataFrame) -> np.ndarray:
    """
    Pull torques for ACTIVE_DOFS from InverseDynamicsTool output.
    Tries column name variants: bare dof name, dof_moment, dof_force.
    """
    n = len(id_df)
    tau = np.zeros((n, len(ACTIVE_DOFS)))
    for j, dof in enumerate(ACTIVE_DOFS):
        for candidate in [dof, dof + "_moment", dof + "_force"]:
            if candidate in id_df.columns:
                tau[:, j] = id_df[candidate].to_numpy()
                break
    return tau


def run_inverse_dynamics(
    model_path: str,
    mot_path: str,
    output_dir: str,
    t_start: float,
    t_end: float,
) -> pd.DataFrame:
    """
    Run OpenSim InverseDynamicsTool on mot_path (no external forces).
    Writes cup_task_id.sto to output_dir and returns it as a DataFrame.
    """
    out_file = "cup_task_id.sto"
    id_tool = osim.InverseDynamicsTool()
    id_tool.setModelFileName(model_path)
    id_tool.setCoordinatesFileName(mot_path)
    id_tool.setResultsDir(output_dir)
    id_tool.setOutputGenForceFileName(out_file)
    id_tool.setStartTime(t_start)
    id_tool.setEndTime(t_end)
    id_tool.setLowpassCutoffFrequency(6.0)   # filter before numerical differentiation
    id_tool.run()
    return _read_sto(os.path.join(output_dir, out_file))


# ===========================================================================
# External-force writers (for F_int cup-task ID)
# ===========================================================================

def _write_ext_forces_mot(path: str, t: np.ndarray,
                          F_xyz: np.ndarray, pos_xyz: np.ndarray) -> None:
    """Write ExternalForce .mot with force vector + point of application."""
    n = len(t)
    with open(path, "w") as f:
        f.write("External_Forces\nversion=1\n")
        f.write(f"nRows={n}\nnColumns=7\ninDegrees=no\n\nendheader\n")
        f.write("time\tF_cup_vx\tF_cup_vy\tF_cup_vz"
                "\tF_cup_px\tF_cup_py\tF_cup_pz\n")
        for i in range(n):
            f.write(f"{t[i]:.6f}\t"
                    f"{F_xyz[i,0]:.6f}\t{F_xyz[i,1]:.6f}\t{F_xyz[i,2]:.6f}\t"
                    f"{pos_xyz[i,0]:.6f}\t{pos_xyz[i,1]:.6f}\t{pos_xyz[i,2]:.6f}\n")


def _write_ext_loads_xml(xml_path: str, mot_path: str) -> None:
    """Write ExternalLoads XML that references the .mot file."""
    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
    <ExternalLoads name="cup_loads">
        <objects>
            <ExternalForce name="F_cup">
                <applied_to_body>{HAND_BODY}</applied_to_body>
                <force_expressed_in_body>ground</force_expressed_in_body>
                <point_expressed_in_body>ground</point_expressed_in_body>
                <force_identifier>F_cup_v</force_identifier>
                <point_identifier>F_cup_p</point_identifier>
                <torque_identifier></torque_identifier>
            </ExternalForce>
        </objects>
        <datafile>{mot_path}</datafile>
    </ExternalLoads>
</OpenSimDocument>
"""
    with open(xml_path, "w") as f:
        f.write(xml)


def _write_id_setup_xml_ext(xml_path: str, results_dir: str, model_file: str,
                             mot_path: str, sto_basename: str,
                             t_start: float, t_end: float,
                             ext_loads_xml: str) -> None:
    """Write InverseDynamicsTool XML with external loads and 6 Hz lowpass."""
    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
    <InverseDynamicsTool name="cup_stiffness_ID">
        <results_directory>{results_dir}</results_directory>
        <model_file>{model_file}</model_file>
        <time_range> {t_start:.6f} {t_end:.6f} </time_range>
        <forces_to_exclude>Muscles</forces_to_exclude>
        <coordinates_file>{mot_path}</coordinates_file>
        <lowpass_cutoff_frequency_for_coordinates>6</lowpass_cutoff_frequency_for_coordinates>
        <external_loads_file>{ext_loads_xml}</external_loads_file>
        <output_gen_force_file>{sto_basename}</output_gen_force_file>
        <joints_to_report_body_forces></joints_to_report_body_forces>
        <output_body_forces_file></output_body_forces_file>
    </InverseDynamicsTool>
</OpenSimDocument>
"""
    with open(xml_path, "w") as f:
        f.write(xml)


def _run_id_with_ext(model_path: str, mot_path: str,
                     sig_df: pd.DataFrame, traj: pd.DataFrame,
                     output_dir: str) -> pd.DataFrame:
    """
    Inverse Dynamics with F_int (cup–hand interaction force) applied as an
    ExternalForce on the hand body — same physics as arm_inverse_dynamics.py
    but using the full 7-DOF ACTIVE_DOFS set.

    F_on_hand = −F_int · d̂   (Newton 3rd law; d̂ = first→last hand motion)
    """
    t     = traj["time"].to_numpy()
    F_int = np.interp(t, sig_df["time"].to_numpy(), sig_df["F_int"].to_numpy())

    # FK pre-pass: hand position at each frame
    print("[stiffness] FK pre-pass (hand positions for ExternalForce point) …")
    model_fk = osim.Model(model_path)
    state_fk = model_fk.initSystem()
    hand_body = model_fk.getBodySet().get(HAND_BODY)
    hand_pos  = np.zeros((len(t), 3))
    for i in range(len(t)):
        row = traj.iloc[i]
        for dof in ACTIVE_DOFS:
            model_fk.updCoordinateSet().get(dof).setValue(state_fk, float(row[dof]))
        model_fk.realizePosition(state_fk)
        p = hand_body.getPositionInGround(state_fk)
        hand_pos[i] = [p.get(k) for k in range(3)]

    d_raw     = hand_pos[-1] - hand_pos[0]
    direction = d_raw / (np.linalg.norm(d_raw) + 1e-8)
    F_on_hand = -F_int[:, None] * direction[None, :]  # (N, 3)

    # Write external-force files
    ext_mot      = os.path.join(output_dir, "cup_task_ext_forces_stiff.mot")
    ext_xml      = os.path.join(output_dir, "cup_task_ext_loads_stiff.xml")
    id_setup_xml = os.path.join(output_dir, "cup_task_id_setup_stiff_ext.xml")
    id_sto_name  = "cup_task_id_stiff_ext.sto"

    _write_ext_forces_mot(ext_mot, t, F_on_hand, hand_pos)
    _write_ext_loads_xml(ext_xml, ext_mot)
    _write_id_setup_xml_ext(
        id_setup_xml,
        results_dir  = output_dir,
        model_file   = model_path,
        mot_path     = mot_path,
        sto_basename = id_sto_name,
        t_start      = float(t[0]),
        t_end        = float(t[-1]),
        ext_loads_xml= ext_xml,
    )

    print("[stiffness] Running InverseDynamicsTool (with F_int) …")
    osim.Logger.setLevel(osim.Logger.Level_Error)
    id_tool = osim.InverseDynamicsTool(id_setup_xml)
    ok = id_tool.run()
    osim.Logger.setLevel(osim.Logger.Level_Info)
    if not ok:
        raise RuntimeError("InverseDynamicsTool (with F_int ext force) returned False")
    return _read_sto(os.path.join(output_dir, id_sto_name))


# ===========================================================================
# Per-frame muscle geometry
# ===========================================================================

def _build_ms_properties(
    model: osim.Model,
    state,
    ms_labels: list,
) -> tuple:
    """
    At the current model state, return:
      R     (n_dof × n_ms)  moment arm matrix [m]
      Fmax  (n_ms,)         max isometric force [N]
      fL    (n_ms,)         force-length multiplier (Gaussian, dimensionless)
      Lopt  (n_ms,)         optimal fiber length [m]
      lm    (n_ms,)         raw fiber length [m]
    """
    muscles = model.getMuscles()
    n_ms  = len(ms_labels)

    Fmax = np.zeros(n_ms)
    fL   = np.zeros(n_ms)
    Lopt = np.zeros(n_ms)
    lm   = np.zeros(n_ms)

    for j, name in enumerate(ms_labels):
        ms = muscles.get(name)
        Fmax[j] = ms.getMaxIsometricForce()
        Lopt[j] = ms.getOptimalFiberLength()

        lm[j]   = ms.getFiberLength(state)
        lm_norm = lm[j] / max(Lopt[j], 1e-6)
        fL[j]   = np.exp(-((lm_norm - 1.0) / FL_GAMMA) ** 2)

    R = compute_moment_arms(model, state, ms_labels, ACTIVE_DOFS)
    return R, Fmax, fL, Lopt, lm


# ===========================================================================
# Jacobian and endpoint stiffness
# ===========================================================================

def _numeric_jacobian(
    model: osim.Model,
    state,
    body_name: str = HAND_BODY,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Numerical Jacobian  d(pos_hand)/d(q)  of shape (3, n_dof)
    via central finite differences.  All DOF values are restored after.
    """
    n_dof = len(ACTIVE_DOFS)
    J     = np.zeros((3, n_dof))

    # Cache current joint angles
    q0 = {dof: model.getCoordinateSet().get(dof).getValue(state)
          for dof in ACTIVE_DOFS}

    body = model.getBodySet().get(body_name)

    for j, dof in enumerate(ACTIVE_DOFS):
        # +ε
        model.updCoordinateSet().get(dof).setValue(state, q0[dof] + eps)
        model.realizePosition(state)
        pp = body.getPositionInGround(state)
        p_plus = np.array([pp.get(k) for k in range(3)])

        # -ε
        model.updCoordinateSet().get(dof).setValue(state, q0[dof] - eps)
        model.realizePosition(state)
        pm = body.getPositionInGround(state)
        p_minus = np.array([pm.get(k) for k in range(3)])

        J[:, j] = (p_plus - p_minus) / (2.0 * eps)

        # Restore
        model.updCoordinateSet().get(dof).setValue(state, q0[dof])

    model.realizePosition(state)
    return J


def _endpoint_stiffness(
    model: osim.Model,
    state,
    a_m: np.ndarray,
    R: np.ndarray,
    Fmax: np.ndarray,
    fL: np.ndarray,
    Lopt: np.ndarray,
    _lm: np.ndarray = None,  # unused; lm already folded into fL
) -> np.ndarray:
    """
    Compute 3×3 endpoint stiffness K_e [N/m] and damping D_e [N·s/m]:

      k_m     = a · Fmax · fL / Lopt              (Hill stiffness slope)
      d_m     = BETA_D · a · Fmax / (V_MAX_FACTOR·Lopt)  (Hill f-v slope)
      K_joint = R  diag(k_m)  Rᵀ                 (joint stiffness)
      D_joint = R  diag(d_m)  Rᵀ                 (joint damping)
      K_e     = J⁺ᵀ  K_joint  J⁺                  (endpoint stiffness)
      D_e     = J⁺ᵀ  D_joint  J⁺                  (endpoint damping)
    """
    Lopt_safe = np.maximum(Lopt, 1e-6)
    k_m     = a_m * Fmax * fL / Lopt_safe                 # (n_ms,)
    d_m     = BETA_D * a_m * Fmax / (V_MAX_FACTOR * Lopt_safe)
    K_joint = R @ np.diag(k_m) @ R.T                      # (n_dof, n_dof)
    D_joint = R @ np.diag(d_m) @ R.T                      # (n_dof, n_dof)
    J       = _numeric_jacobian(model, state)             # (3, n_dof)
    Jpinv   = np.linalg.pinv(J)                           # (n_dof, 3)
    K_e     = Jpinv.T @ K_joint @ Jpinv                   # (3, 3)
    D_e     = Jpinv.T @ D_joint @ Jpinv                   # (3, 3)
    return K_e, K_joint, D_e, D_joint


# ===========================================================================
# Main stiffness pipeline
# ===========================================================================

def run_stiffness(
    traj: pd.DataFrame,
    perturb_mask: np.ndarray,
    model_path: str,
    ms_labels: list,
    mode: str = "static_opt",
    sig_df: pd.DataFrame = None,
) -> tuple:
    """
    Compute endpoint stiffness K_e(t) for every frame in traj.
    Also collects normalized fiber lengths in the same model pass
    (avoids a second equilibrateMuscles pass).

    Parameters
    ----------
    traj          : joint angle DataFrame from build_trajectory (+ perturbations)
    perturb_mask  : boolean array, True during perturbation windows
    model_path    : path to MOBL_ARMS_41.osim
    ms_labels     : list of muscle names (subset of model muscles)
    mode          : 'static_opt' or 'cmc'
    sig_df        : optional signals DataFrame (from arm_cup_perturbation.py).
                    If it contains an 'F_int' column the ID will include the
                    cup–hand interaction force as an ExternalForce on the hand.
                    This gives physically correct joint torques for the cup task.

    Returns
    -------
    stiff_df : DataFrame with columns:
                 time, perturb, p_null,
                 a_<muscle> for each muscle,
                 K_e_xx, K_e_yy, K_e_zz, K_e_xy, K_e_xz, K_e_yz,
                 K_min, K_max  (eigenvalues of 2-D XY stiffness ellipse)
    ms_lens  : DataFrame with columns: time, <muscle>...  (normalised fiber lengths)
    """
    print(f"[stiffness] mode = {mode}")

    # Write perturbed trajectory as .mot for InverseDynamicsTool
    mot_p = os.path.join(OUTPUT_DIR, "cup_task_traj_stiff.mot")
    write_mot(traj, mot_p)

    # ── Inverse Dynamics ──────────────────────────────────────────────────
    t0 = float(traj["time"].iloc[0])
    tf = float(traj["time"].iloc[-1])

    use_ext = sig_df is not None and "F_int" in sig_df.columns
    if use_ext:
        print("[stiffness] Running InverseDynamicsTool with F_int external wrench …")
        id_df = _run_id_with_ext(model_path, mot_p, sig_df, traj, OUTPUT_DIR)
    else:
        print("[stiffness] Running InverseDynamicsTool (kinematics only) …")
        id_df = run_inverse_dynamics(model_path, mot_p, OUTPUT_DIR, t0, tf)

    tau_raw  = _extract_active_torques(id_df)               # (N_id, n_dof)

    # Interpolate ID torques onto the trajectory time grid
    t_id   = id_df.iloc[:, 0].to_numpy()
    t_traj = traj["time"].to_numpy()
    tau    = np.stack(
        [sp_interp1d(t_id, tau_raw[:, j], bounds_error=False, fill_value=0.0)(t_traj)
         for j in range(len(ACTIVE_DOFS))],
        axis=1,
    )   # (N_traj, n_dof)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"[stiffness] Loading model: {model_path}")
    model = osim.Model(model_path)
    state = model.initSystem()

    # Optimal fiber lengths (l_opt) for normalisation — model constants, no state needed
    muscles  = model.getMuscles()
    l_opt = {ms: muscles.get(ms).getOptimalFiberLength() for ms in ms_labels}

    # ── Smooth co-contraction floor (replaces hard step) ─────────────────
    # CMC mode: Gaussian ramp centred on T_PERT, peak = COCONTRACTION_ALPHA,
    # baseline = BASELINE_ALPHA. Width SIGMA_FLOOR > SIGMA_PERT so the floor
    # ramps in/out continuously rather than as a square wave — this kills
    # the block-like jumps in K(t) and matches the smooth pre-activation
    # observed in real perturbation EMG.
    t_traj_arr = traj["time"].to_numpy()
    if mode == "cmc" and perturb_mask.any():
        ramp = np.exp(-0.5 * ((t_traj_arr - T_PERT) / SIGMA_FLOOR) ** 2)
        floor_t = BASELINE_ALPHA + (COCONTRACTION_ALPHA - BASELINE_ALPHA) * ramp
    else:
        floor_t = np.full_like(t_traj_arr, BASELINE_ALPHA)

    # First-order activation low-pass coefficient.
    # a_filt[i] = α·a_raw[i] + (1-α)·a_filt[i-1]; α = dt/(τ + dt)
    dt_traj = 1.0 / float(FS)
    ema_alpha = dt_traj / (TAU_ACT + dt_traj)

    # ── Frame-by-frame loop (fiber lengths + stiffness in one pass) ───────
    stiff_records = []
    flen_records  = []
    a0       = np.full(len(ms_labels), BASELINE_ALPHA)   # SLSQP warm-start
    a_filt   = np.full(len(ms_labels), BASELINE_ALPHA)   # filtered activations (state)

    for i in tqdm(range(len(traj)), desc=f"stiffness+fibers/{mode}"):
        row = traj.iloc[i]
        alpha_floor = float(floor_t[i])

        # Set joint angles
        for dof in ACTIVE_DOFS:
            model.updCoordinateSet().get(dof).setValue(state, float(row[dof]))
        model.realizePosition(state)
        model.equilibrateMuscles(state)

        # Muscle geometry at this posture (includes raw fiber lengths lm)
        R, Fmax, fL, Lopt, lm = _build_ms_properties(model, state, ms_labels)

        # ── Fiber lengths (normalised) ────────────────────────────────────
        frec = {"time": float(row["time"])}
        for j, ms in enumerate(ms_labels):
            frec[ms] = lm[j] / max(l_opt[ms], 1e-6)
        flen_records.append(frec)

        # ── Static optimization → raw activations (Crowninshield–Brand) ──
        a_raw = so_frame(tau[i], R, Fmax, fL,
                         alpha_floor=alpha_floor, x0=a0, cost_exp=3)
        a0    = a_raw.copy()                 # warm start for next SO frame

        # ── Activation dynamics low-pass (EMD + activation/deactivation) ─
        # Use filtered activations for stiffness/damping/output. This
        # eliminates the unphysical square-wave activations that would
        # otherwise propagate into K_e(t) and D_e(t).
        a_filt = ema_alpha * a_raw + (1.0 - ema_alpha) * a_filt
        a_filt = np.maximum(a_filt, alpha_floor)   # respect time-varying floor
        a_m    = a_filt

        # ── Endpoint stiffness + damping (3×3) + joint counterparts ──────
        K_e, K_joint, D_e, D_joint = _endpoint_stiffness(
            model, state, a_m, R, Fmax, fL, Lopt
        )

        # Co-contraction proxy P_null = min activation (null-space floor)
        # This is the signal that drives c_exo in the tele-impedance framework.
        p_null = float(np.min(a_m))

        rec = {
            "time":    float(row["time"]),
            "perturb": bool(perturb_mask[i]),
            "p_null":  p_null,
        }
        for j, ms in enumerate(ms_labels):
            rec[f"a_{ms}"] = float(a_m[j])
        for (r, c), key in [
            ((0, 0), "K_e_xx"), ((1, 1), "K_e_yy"), ((2, 2), "K_e_zz"),
            ((0, 1), "K_e_xy"), ((0, 2), "K_e_xz"), ((1, 2), "K_e_yz"),
        ]:
            rec[key] = float(K_e[r, c])

        eigs = np.linalg.eigvalsh(K_e[:2, :2])
        rec["K_min"] = float(eigs[0])
        rec["K_max"] = float(eigs[1])
        # Endpoint damping (3×3)
        for (r, c), key in [
            ((0, 0), "D_e_xx"), ((1, 1), "D_e_yy"), ((2, 2), "D_e_zz"),
            ((0, 1), "D_e_xy"), ((0, 2), "D_e_xz"), ((1, 2), "D_e_yz"),
        ]:
            rec[key] = float(D_e[r, c])
        deigs = np.linalg.eigvalsh(D_e[:2, :2])
        rec["D_min"] = float(deigs[0])
        rec["D_max"] = float(deigs[1])
        # Joint stiffness diagonal (N·m/rad per DOF)
        for di, dof in enumerate(ACTIVE_DOFS):
            rec[f"K_{dof}"] = float(K_joint[di, di])
            rec[f"D_{dof}"] = float(D_joint[di, di])
        if len(ACTIVE_DOFS) >= 2:
            rec["K_cross"] = float(K_joint[0, 1])
        stiff_records.append(rec)

    return pd.DataFrame(stiff_records), pd.DataFrame(flen_records)


# ===========================================================================
# Stiffness plot
# ===========================================================================

def plot_stiffness(stiff_df: pd.DataFrame, save_path: str, mode: str = ""):
    t  = stiff_df["time"].to_numpy()
    pm = stiff_df["perturb"].to_numpy().astype(bool)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # ── Panel 1: diagonal endpoint stiffness ──────────────────────────────
    axes[0].plot(t, stiff_df["K_e_xx"], label="K_xx")
    axes[0].plot(t, stiff_df["K_e_yy"], label="K_yy")
    axes[0].set_ylabel("K_e diagonal (N/m)")
    axes[0].legend(fontsize=8)

    # ── Panel 2: stiffness ellipse eigenvalues ────────────────────────────
    axes[1].plot(t, stiff_df["K_min"], label="λ_min")
    axes[1].plot(t, stiff_df["K_max"], label="λ_max")
    axes[1].set_ylabel("K_e eigenvalues (N/m)")
    axes[1].legend(fontsize=8)

    # ── Panel 3: representative activations + P_null ───────────────────────
    for ms in ["a_BIClong", "a_TRIlong", "a_DELT1"]:
        if ms in stiff_df.columns:
            axes[2].plot(t, stiff_df[ms], label=ms.replace("a_", ""))
    axes[2].plot(t, stiff_df["p_null"], color="k", linestyle="--", label="P_null → c_exo")
    axes[2].set_ylabel("Activation")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(fontsize=8)

    # ── Shade perturbation windows ─────────────────────────────────────────
    for ax in axes:
        ax.grid(True, alpha=0.3)
        if pm.any():
            ylim = ax.get_ylim()
            ax.fill_between(t, ylim[0], ylim[1], where=pm,
                            alpha=0.15, color="red", zorder=0)
            ax.set_ylim(ylim)

    fig.suptitle(f"Endpoint stiffness modulation — {mode}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved stiffness plot: {save_path}")


def plot_cocontraction_stiffness(stiff_df: pd.DataFrame, save_path: str,
                                mode: str = "") -> None:
    """
    3-panel plot (mirrors arm_inverse_dynamics.py output):
      1. Elbow flexor + extensor activations from SO
      2. Co-contraction index  CCI = 2·min(A_flex, A_ext) / (A_flex + A_ext)
      3. Joint stiffness K_elbow_flexion and K_elv_angle  [N·m/rad]
    """
    t  = stiff_df["time"].to_numpy()
    pm = stiff_df["perturb"].to_numpy().astype(bool)
    cmap = matplotlib.colormaps["tab10"]

    flex_ms = [m for m in ["BIClong", "BICshort", "BRA", "BRD"]
               if f"a_{m}" in stiff_df.columns]
    ext_ms  = [m for m in ["TRIlong", "TRIlat", "TRImed"]
               if f"a_{m}" in stiff_df.columns]

    A_flex = (sum(stiff_df[f"a_{m}"].to_numpy() for m in flex_ms)
              if flex_ms else np.zeros(len(t)))
    A_ext  = (sum(stiff_df[f"a_{m}"].to_numpy() for m in ext_ms)
              if ext_ms  else np.zeros(len(t)))
    CCI    = 2.0 * np.minimum(A_flex, A_ext) / (A_flex + A_ext + 1e-9)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # ── Panel 1: activations ──────────────────────────────────────────────
    ax1 = axes[0]
    for idx, ms in enumerate(flex_ms):
        ax1.plot(t, stiff_df[f"a_{ms}"], color=cmap(idx), linewidth=1.5,
                 label=f"{ms} (flex)")
    for idx, ms in enumerate(ext_ms):
        ax1.plot(t, stiff_df[f"a_{ms}"], color=cmap(idx + 4), linewidth=1.5,
                 linestyle="--", label=f"{ms} (ext)")
    if pm.any():
        ax1.axvline(t[pm][0], color="red", linewidth=0.9, linestyle=":",
                    alpha=0.8, label="Perturbation start")
    ax1.set_ylabel("Activation (SO, filtered)", fontsize=9)
    ax1.set_title("Model-inferred elbow muscle activations  "
                  "(Crowninshield\u2013Brand SO + 40 ms activation dynamics)",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=7, ncol=4, loc="upper right")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.25)

    # ── Panel 2: CCI ──────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.fill_between(t, CCI, alpha=0.35, color="#E91E63")
    ax2.plot(t, CCI, color="#E91E63", linewidth=2.0, label="CCI elbow")
    if pm.any():
        ax2.axvline(t[pm][0], color="red", linewidth=0.9, linestyle=":", alpha=0.8)
    ax2.set_ylabel("CCI  (0\u20131)", fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Co-contraction index  CCI = 2\u00b7min(A_flex, A_ext) / (A_flex + A_ext)",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)

    # ── Panel 3: joint stiffness ──────────────────────────────────────────
    ax3 = axes[2]
    for col, clr, lbl in [
        ("K_elbow_flexion", "#2196F3", "K_elbow_flex  (N\u00b7m/rad)"),
        ("K_elv_angle",     "#FF5722", "K_elv_angle   (N\u00b7m/rad)"),
    ]:
        if col in stiff_df.columns:
            ax3.plot(t, stiff_df[col], color=clr, linewidth=1.8, label=lbl)
    if pm.any():
        ax3.axvline(t[pm][0], color="red", linewidth=0.9, linestyle=":",
                    alpha=0.8, label="Perturbation start")

    # Shade perturbation windows across all panels
    for ax in axes:
        if pm.any():
            ylim = ax.get_ylim()
            ax.fill_between(t, ylim[0], ylim[1], where=pm,
                            alpha=0.10, color="red", zorder=0)
            ax.set_ylim(ylim)

    ax3.set_ylabel("Joint stiffness (N\u00b7m/rad)", fontsize=9)
    ax3.set_xlabel("Time (s)", fontsize=10)
    ax3.set_title("Model-inferred joint stiffness  K = R · diag(a·F₀·fl/ℓ_opt) · Rᵀ",
                  fontsize=10, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.25)

    fig.suptitle(
        f"Model-inferred co-contraction and joint stiffness  "
        f"(SO proxy, not measured stiffness)  — {mode}",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ===========================================================================
# Class wrapper — ComputeStiffness
# ===========================================================================
class ComputeStiffness:
    """Object-oriented driver for the endpoint-stiffness computation pipeline.

    Encapsulates trajectory loading, optional perturbation injection, ID +
    SO with co-contraction floor, K_e/D_e tensors, and plotting.
    """

    def __init__(self,
                 traj_mode:  str = "perturb",
                 stiff_mode: str = "none",
                 do_perturb: bool = False,
                 model_path: str = MODEL_PATH,
                 ms_labels:  list = None,
                 output_dir: str = OUTPUT_DIR):
        if stiff_mode == "cmc":
            do_perturb = True
        self.traj_mode  = traj_mode
        self.stiff_mode = stiff_mode
        self.do_perturb = do_perturb
        self.model_path = model_path
        self.ms_labels  = list(ms_labels) if ms_labels is not None else list(MS_LABELS)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ── End-to-end ────────────────────────────────────────────────────────
    def run(self) -> dict:
        traj_mode  = self.traj_mode
        stiff_mode = self.stiff_mode
        do_perturb = self.do_perturb

        print(f"Trajectory  : {traj_mode}")
        print(f"Stiffness   : {stiff_mode}")

        # ── 1. Build / load trajectory + perturb_mask ─────────────────────
        if traj_mode == "perturb":
            from arm_cup_perturbation import (
                T_PERT,
                OUTPUT_DIR as _PERT_OUT_DIR,
                TAG        as _PERT_TAG,
            )
            _traj_csv = os.path.join(_PERT_OUT_DIR, f"cup_task_trajectory_{_PERT_TAG}.csv")
            _sig_csv  = os.path.join(_PERT_OUT_DIR, f"cup_task_signals_{_PERT_TAG}.csv")
            if not os.path.exists(_traj_csv):
                raise FileNotFoundError(
                    f"Run arm_cup_perturbation.py first to generate:\n  {_traj_csv}")
            traj = pd.read_csv(_traj_csv)
            sig  = pd.read_csv(_sig_csv)
            perturb_mask = sig["F_pert"].to_numpy() > 1.0  # F_pert is force spike (N)
            tag = "perturb"
            if stiff_mode != "none":
                tag += f"_{stiff_mode}"
            print(f"Loaded trajectory : {_traj_csv}  ({len(traj)} frames @ {FS} Hz)")
            print(f"Perturbation window: {perturb_mask.sum()} frames "
                  f"(F_pert > 1.0 N, centred at t = {T_PERT:.2f} s)")
            if stiff_mode != "none" and "F_int" in sig.columns:
                print("  F_int column found — ID will include cup-hand interaction force.")
        else:
            tag = traj_mode
            if do_perturb:
                tag += "_perturbed"
            if stiff_mode != "none":
                tag += f"_{stiff_mode}"

            if traj_mode == "linear":
                from arm_linear_trajectory import build_linear_trajectory, _init_model as _lin_init
                _lin_model, _lin_state = _lin_init()
                traj, *_ = build_linear_trajectory(_lin_model, _lin_state)
            else:
                traj = build_trajectory(traj_mode)

            perturb_mask = np.zeros(len(traj), dtype=bool)
            sig = None
            if do_perturb:
                traj, perturb_mask = add_perturbations(traj)
                print(f"Perturbations injected at t = {PERTURB_TIMES} s "
                      f"(dur={PERTURB_DURATION} s, mag={PERTURB_MAG_DEG}°)")

        # ── 3. Save trajectory + write .mot ───────────────────────────────
        traj_csv = os.path.join(self.output_dir, f"cup_task_trajectory_{tag}.csv")
        traj.to_csv(traj_csv, index=False)
        print(f"Trajectory saved: {traj_csv}  ({len(traj)} frames @ {FS} Hz)")

        mot_path = os.path.join(self.output_dir, f"cup_task_trajectory_{tag}.mot")
        write_mot(traj, mot_path)
        plot_trajectory(traj, os.path.join(self.output_dir, f"cup_task_trajectory_{tag}.png"))

        # ── 4. Fiber lengths (+ stiffness if requested) ───────────────────
        if stiff_mode != "none":
            _sig_for_id = sig if (traj_mode == "perturb") else None
            stiff_df, ms_lens = run_stiffness(
                traj, perturb_mask, self.model_path, self.ms_labels,
                mode=stiff_mode, sig_df=_sig_for_id,
            )
        else:
            ms_lens  = run_msk(traj, self.model_path, self.ms_labels, normalise=True)
            stiff_df = None

        ms_lens.to_csv(
            os.path.join(self.output_dir, f"cup_task_fiber_lengths_{tag}.csv"), index=False)
        plot_fiber_lengths(
            ms_lens, os.path.join(self.output_dir, f"cup_task_fiber_lengths_{tag}.png"))

        out = {"traj_csv": traj_csv, "mot": mot_path}

        # ── 5. Stiffness output ───────────────────────────────────────────
        if stiff_df is not None:
            stiff_csv = os.path.join(self.output_dir, f"cup_task_stiffness_{tag}.csv")
            stiff_df.to_csv(stiff_csv, index=False)
            print(f"Stiffness saved : {stiff_csv}")
            plot_stiffness(stiff_df,
                os.path.join(self.output_dir, f"cup_task_stiffness_{tag}.png"),
                mode=stiff_mode)
            plot_cocontraction_stiffness(stiff_df,
                os.path.join(self.output_dir, f"cup_task_cocontraction_stiffness_{tag}.png"),
                mode=stiff_mode)
            out["stiff_csv"] = stiff_csv
            print(
                "\nNext steps:\n"
                "  1. Feed stiffness CSV into NeuroMotion (scripts/mov2emg.py) for\n"
                "     synthetic EMG synthesis modulated by K_e(t).\n"
                "  2. Use p_null column as c_exo setpoint for the SEA exo cable model.\n"
                "  3. Use K_e(t) as the reward shaping signal in rl/train_ppo_residual.py."
            )
        else:
            print("\nNo stiffness computed. Re-run with --stiffness static_opt or --stiffness cmc.")
        return out


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MoBL-ARMS cup-task stiffness computation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python compute_stiffness.py --mode perturb --stiffness static_opt\n"
            "  python compute_stiffness.py --mode perturb --stiffness cmc\n"
            "  python compute_stiffness.py --mode sine --stiffness static_opt --perturb\n"
            "  python compute_stiffness.py --mode min_jerk --stiffness static_opt --perturb\n"
        ),
    )
    parser.add_argument(
        "--mode", choices=["sine", "min_jerk", "linear", "perturb"], default="perturb",
        help="Trajectory mode (default: perturb).")
    parser.add_argument(
        "--stiffness", choices=["none", "static_opt", "cmc"], default="cmc",
        help="Stiffness computation mode (default: cmc).")
    parser.add_argument(
        "--perturb", action="store_true",
        help="Add shoulder-rotation bumps. Auto-enabled for --stiffness cmc.")
    args = parser.parse_args()

    ComputeStiffness(
        traj_mode  = args.mode,
        stiff_mode = args.stiffness,
        do_perturb = args.perturb,
    ).run()
