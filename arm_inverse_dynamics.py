"""
arm_inverse_dynamics.py

Inverse Dynamics for the perturbed cup trajectory.
F_int is applied as an external wrench on the hand body.

Pipeline
--------
  1.  Load  q(t)     from  cup_task_trajectory_perturb.csv
      Load  F_int(t) from  cup_task_signals_perturb.csv

  2.  Fit CubicSpline to q → analytical qdot(t), qddot(t)
      (avoids noise amplification that np.gradient would introduce)

  3.  FK all frames  → hand_pos(t) for the point of application.
      Finite-difference Jacobian J(q) ∈ ℝ^{3×2} for endpoint-force recovery.

  4.  Build external wrench on the hand (Newton's 3rd law):

          F_on_hand(t) = −F_int(t) · d̂

      where d̂ is the unit direction of motion in ground frame.

      Why include the inertial baseline M_eff × a_mj(t)?
      ─────────────────────────────────────────────────────
        •  F_int = M_eff·a_mj  +  F_spike  −  F_under
        •  M_eff·a_mj  is the cup+ball inertia reacting against the hand.
        •  The cup is NOT modelled as a rigid body in OpenSim, so its
           inertial load must enter as an external force on the hand.
        •  The arm's own inertia (M_arm) is handled internally by
           OpenSim's rigid-body equations.  Do NOT add it here.

  5.  Write  ext_forces.mot  +  ext_loads.xml  +  ID setup XML.

  6.  Run  osim.InverseDynamicsTool  → raw joint torques  τ(t)  in a .sto.

  7.  Parse τ; run a second baseline ID (no external force) → τ₀(t).

  8.  Recover F_int via directional scalar projection:

          Δτ = τ − τ₀  =  F_int · (J^T d̂)   [from the ID equation]

          j_line = J^T d̂  ∈ ℝ²

          F_int_recovered = (j_line · Δτ) / (j_line · j_line)

      This closes the loop exactly because only the external-force
      contribution is isolated before inverting.

  8.  Save results CSV + two plots.

Does q follow the prescribed angles?
──────────────────────────────────────
  YES by construction.  The CSV from arm_cup_perturbation.py is the IK
  solution; it IS q(t).  ID tells us which τ(t) are *required* to achieve
  that motion.  If τ exceeds muscle capacity the arm would deviate in
  reality – the plot of τ vs time shows exactly where that risk is highest
  (the spike at t = T_PERT).

Outputs  (demo_output/)
-----------------------
  cup_task_ext_forces_perturb.mot   – external force time series
  cup_task_ext_loads_perturb.xml    – ExternalLoads XML
  cup_task_id_setup_perturb.xml     – InverseDynamicsTool setup XML
  cup_task_id_perturb.sto           – raw ID generalised-force output
  cup_task_id_baseline_perturb.sto  – baseline ID (no ext force)
  cup_task_id_results_perturb.csv   – τ(t), τ₀(t), F_int_recovered(t)
  cup_task_id_torques_perturb.png   – joint torques vs time
  cup_task_id_endpoint_perturb.png  – F_int_recovered vs F_int prescribed

Usage
-----
  python arm_inverse_dynamics.py
  python arm_inverse_dynamics.py --tag perturb
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
from scipy.interpolate import CubicSpline, interp1d
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_cup_task import (MODEL_PATH, DEMO_OUTPUT_DIR, ACTIVE_DOFS, NEUTRAL, MS_LABELS,
                          compute_moment_arms, so_frame)
import os as _os
SCRIPT_NAME   = _os.path.splitext(_os.path.basename(__file__))[0]
OUTPUT_DIR    = _os.path.join(DEMO_OUTPUT_DIR, SCRIPT_NAME)
PERTURB_DIR   = _os.path.join(DEMO_OUTPUT_DIR, "arm_cup_perturbation")
_os.makedirs(OUTPUT_DIR, exist_ok=True)
from arm_cup_perturbation import (
    LOCKED, HAND_BODY, HAND_LOCAL_PT, T_PERT, T_MOVE,
    ELV_ANGLE_BOUNDS, ELBOW_FLEX_BOUNDS,
)

from config import BASELINE_ALPHA, ACTIVE_IK_DOFS

# ===========================================================================
# Constants
# ===========================================================================
TAG      = "perturb"
EPS_JAC  = 1e-6       # rad  – finite-difference step for Jacobian
# BASELINE_ALPHA and ACTIVE_IK_DOFS imported from config.py

# ===========================================================================
# FK + Jacobian
# ===========================================================================

def _init_model() -> tuple:
    m = osim.Model(MODEL_PATH)
    m.setUseVisualizer(False)
    s = m.initSystem()
    for dof, val in LOCKED.items():
        m.updCoordinateSet().get(dof).setValue(s, val, False)
    m.assemble(s)
    return m, s


def _hand_pos(model, state, q: np.ndarray) -> np.ndarray:
    """FK: (elv_angle, elbow_flexion) = q  →  hand position in ground."""
    model.updCoordinateSet().get("elv_angle").setValue(state, q[0], False)
    model.updCoordinateSet().get("elbow_flexion").setValue(state, q[1], False)
    model.realizePosition(state)
    p = model.getBodySet().get(HAND_BODY).findStationLocationInGround(
        state, osim.Vec3(*HAND_LOCAL_PT))
    return np.array([p[0], p[1], p[2]])


def compute_jacobian(model, state, q: np.ndarray) -> np.ndarray:
    """
    Central-difference Jacobian  J ∈ ℝ^{3×2}:

        J_ij = ∂p_i / ∂q_j  ≈  [ FK(q + ε·eⱼ) − FK(q − ε·eⱼ) ] / (2ε)
    """
    J = np.zeros((3, 2))
    for j in range(2):
        dq    = np.zeros(2); dq[j] = EPS_JAC
        J[:, j] = (_hand_pos(model, state, q + dq)
                   - _hand_pos(model, state, q - dq)) / (2 * EPS_JAC)
    return J


# ===========================================================================
# File writers
# ===========================================================================

def write_ext_forces_mot(path: str,
                         t: np.ndarray,
                         F_xyz: np.ndarray,
                         pos_xyz: np.ndarray) -> None:
    """
    Write OpenSim external-force .mot file.

    Columns
    -------
    time | F_perturb_vx/vy/vz  (force expressed in ground)
         | F_perturb_px/py/pz  (point expressed in ground)

    The column prefix  F_perturb_v  and  F_perturb_p  must match the
    force_identifier / point_identifier in the ExternalLoads XML.
    """
    n = len(t)
    with open(path, "w") as f:
        f.write("External_Forces\n")
        f.write("version=1\n")
        f.write(f"nRows={n}\n")
        f.write("nColumns=7\n")
        f.write("inDegrees=no\n")
        f.write("\n")
        f.write("endheader\n")
        f.write("time\tF_perturb_vx\tF_perturb_vy\tF_perturb_vz"
                "\tF_perturb_px\tF_perturb_py\tF_perturb_pz\n")
        for i in range(n):
            f.write(
                f"{t[i]:.6f}\t"
                f"{F_xyz[i, 0]:.6f}\t{F_xyz[i, 1]:.6f}\t{F_xyz[i, 2]:.6f}\t"
                f"{pos_xyz[i, 0]:.6f}\t{pos_xyz[i, 1]:.6f}\t{pos_xyz[i, 2]:.6f}\n"
            )


def write_ext_loads_xml(xml_path: str, mot_path: str) -> None:
    """
    ExternalLoads XML consumed by InverseDynamicsTool.

    force_expressed_in_body = ground  →  F_xyz columns are in ground frame.
    point_expressed_in_body = ground  →  pos_xyz columns are in ground frame.
    """
    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
    <ExternalLoads name="perturbation_loads">
        <objects>
            <ExternalForce name="F_perturb">
                <applied_to_body>hand</applied_to_body>
                <force_expressed_in_body>ground</force_expressed_in_body>
                <point_expressed_in_body>ground</point_expressed_in_body>
                <force_identifier>F_perturb_v</force_identifier>
                <point_identifier>F_perturb_p</point_identifier>
                <torque_identifier></torque_identifier>
            </ExternalForce>
        </objects>
        <datafile>{mot_path}</datafile>
    </ExternalLoads>
</OpenSimDocument>
"""
    with open(xml_path, "w") as f:
        f.write(xml)


def write_id_setup_xml(xml_path: str,
                       results_dir: str,
                       model_file: str,
                       mot_path: str,
                       sto_basename: str,
                       t_start: float,
                       t_end: float,
                       ext_loads_xml: str = "") -> None:
    """
    InverseDynamicsTool setup XML.  All paths are absolute so the tool
    can be run from any working directory.
    Pass ext_loads_xml="" (default) to run without external forces (baseline).
    """
    ext_line = (f"        <external_loads_file>{ext_loads_xml}</external_loads_file>\n"
                if ext_loads_xml else "")
    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
    <InverseDynamicsTool name="cup_ID_perturb">
        <results_directory>{results_dir}</results_directory>
        <model_file>{model_file}</model_file>
        <time_range> {t_start:.6f} {t_end:.6f} </time_range>
        <forces_to_exclude>Muscles</forces_to_exclude>
        <coordinates_file>{mot_path}</coordinates_file>
        <lowpass_cutoff_frequency_for_coordinates>6</lowpass_cutoff_frequency_for_coordinates>
{ext_line}        <output_gen_force_file>{sto_basename}</output_gen_force_file>
        <joints_to_report_body_forces></joints_to_report_body_forces>
        <output_body_forces_file></output_body_forces_file>
    </InverseDynamicsTool>
</OpenSimDocument>
"""
    with open(xml_path, "w") as f:
        f.write(xml)


# ===========================================================================
# .sto parser
# ===========================================================================

def read_sto(path: str) -> pd.DataFrame:
    """Parse an OpenSim .sto file → DataFrame."""
    with open(path) as f:
        lines = f.readlines()
    header_end = next(i for i, l in enumerate(lines) if l.strip() == "endheader")
    cols = lines[header_end + 1].strip().split("\t")
    data = [list(map(float, l.split()))
            for l in lines[header_end + 2:] if l.strip()]
    return pd.DataFrame(data, columns=cols)


def _find_col(df: pd.DataFrame, keyword: str) -> str | None:
    """Return first column whose name contains keyword (case-insensitive)."""
    hits = [c for c in df.columns if keyword.lower() in c.lower()]
    return hits[0] if hits else None


# ===========================================================================
# Static Optimisation helpers
# ===========================================================================


def run_active_msk(traj: pd.DataFrame,
                   tau_on_traj: np.ndarray,
                   model_path: str,
                   ms_labels: list,
                   active_ik_dofs: list) -> pd.DataFrame:
    """
    Activation-corrected normalised fiber lengths, per-muscle activations,
    and joint stiffness.

    At each timestep:
      1. Set q from traj.
      2. Compute moment arms R and force–length multipliers fl.
      3. Static optimisation → activations a(t).
      4. Set activations + equilibrateMuscles → getFiberLength.
      5. Compute joint stiffness  K = R · diag(k_m) · R^T
         where  k_m_i = BETA · a_i · F0_i · fl_i / l_opt_i.

    Normalisation of fiber lengths: divided by neutral-posture passive length.

    Parameters
    ----------
    tau_on_traj : (N, n_active_ik_dofs)  joint torques aligned to traj.time

    Returns
    -------
    fl_df  : DataFrame  time + normalised fiber length per muscle
    act_df : DataFrame  time + activation per muscle
    K_df   : DataFrame  time + K_<dof> for each active_ik_dof + K_cross
    """
    BETA = 40.0   # Hill model dimensionless stiffness coefficient

    model = osim.Model(model_path)
    model.setUseVisualizer(False)
    state = model.initSystem()

    # Locked DOFs
    for dof, val in LOCKED.items():
        model.updCoordinateSet().get(dof).setValue(state, val, False)
    model.assemble(state)

    # Model constants
    F0    = np.array([model.getMuscles().get(ms).getMaxIsometricForce()
                      for ms in ms_labels])
    l_opt = np.array([model.getMuscles().get(ms).getOptimalFiberLength()
                      for ms in ms_labels])
    l_opt = np.clip(l_opt, 1e-4, None)

    # Diagnostic: neutral-pose passive fiber length vs l_opt
    # (flags biarticular muscles like TRIlong that are far from optimal at neutral)
    for dof in ACTIVE_DOFS:
        model.updCoordinateSet().get(dof).setValue(state, NEUTRAL[dof], False)
    model.assemble(state)
    model.equilibrateMuscles(state)
    print(f"  {'Muscle':<12}  {'l_neutral (m)':>14}  {'l_opt (m)':>10}  {'ratio':>7}")
    for j_ms, ms in enumerate(ms_labels):
        l_n = model.getMuscles().get(ms).getFiberLength(state)
        lo  = l_opt[j_ms]
        flag = "  ← FAR FROM OPTIMAL" if abs(l_n / lo - 1.0) > 0.35 else ""
        print(f"  {ms:<12}  {l_n:>14.4f}  {lo:>10.4f}  {l_n/lo:>7.3f}{flag}")

    n_dofs = len(active_ik_dofs)
    fl_records  = []
    act_records = []
    K_records   = []

    for i in tqdm(range(len(traj)), desc="Active FL + Stiffness (SO)", ncols=80):
        row = traj.iloc[i]
        tau_i = tau_on_traj[i]           # (n_active_ik_dofs,)

        # Set all DOFs to the trajectory pose
        for dof in ACTIVE_DOFS:
            model.updCoordinateSet().get(dof).setValue(state, row[dof], False)
        model.realizeVelocity(state)     # needed for force–length & moment arms

        # Force–length multipliers at this pose
        fl_mult = np.array([
            model.getMuscles().get(ms).getActiveForceLengthMultiplier(state)
            for ms in ms_labels
        ])
        fl_mult = np.clip(fl_mult, 1e-4, None)

        # Moment arm matrix  R ∈ ℝ^{n_dofs × n_ms}
        R = compute_moment_arms(model, state, ms_labels, active_ik_dofs)

        # Static optimisation → activations
        # alpha_floor=BASELINE_ALPHA keeps every muscle continuously active
        # so the activation trace can drive NeuroMotion EMG synthesis.
        a = so_frame(tau_i, R, F0, fl_mult, alpha_floor=BASELINE_ALPHA)

        # Set activations, re-equilibrate, read fiber lengths
        for j, ms_name in enumerate(ms_labels):
            model.getMuscles().get(ms_name).setActivation(state, float(a[j]))
        model.equilibrateMuscles(state)

        # ── Fiber lengths normalised by l_opt ──
        fl_rec = {"time": row["time"]}
        for j_ms, ms in enumerate(ms_labels):
            fl_rec[ms] = model.getMuscles().get(ms).getFiberLength(state) / l_opt[j_ms]
        fl_records.append(fl_rec)

        # ── Activations ──
        act_rec = {"time": row["time"]}
        for j_ms, ms in enumerate(ms_labels):
            act_rec[ms] = float(a[j_ms])
        act_records.append(act_rec)

        # ── Joint stiffness  K = R · diag(k_m) · R^T ──
        #   k_m_i = BETA · a_i · F0_i · fl_i / l_opt_i   [N/m]
        k_m   = BETA * a * F0 * fl_mult / l_opt           # (n_ms,)
        K_mat = R @ np.diag(k_m) @ R.T                    # (n_dofs, n_dofs)  [N·m/rad]

        # ── Endpoint (Cartesian) stiffness  K_e = J⁺ᵀ · K_joint · J⁺ ──
        #   Tele-Impedance Stage-2 (Ajoudani et al. 2012) calibrates exactly
        #   this quantity from EMG.  λ_min, λ_max are the principal stiffness
        #   axes of the endpoint stiffness ellipse.
        q_ea_i  = np.array([row["elv_angle"], row["elbow_flexion"]])
        J_i     = compute_jacobian(model, state, q_ea_i)   # (3, 2)
        Jpinv   = np.linalg.pinv(J_i)                       # (2, 3)
        K_e     = Jpinv.T @ K_mat @ Jpinv                   # (3, 3)  [N/m]
        # XY-plane ellipse eigenvalues (cup task is planar in elevation plane)
        eigs    = np.linalg.eigvalsh(K_e[:2, :2])
        lam_min, lam_max = float(eigs[0]), float(eigs[1])

        # restore active-DOF angles (compute_jacobian perturbed them)
        for dof in ACTIVE_DOFS:
            model.updCoordinateSet().get(dof).setValue(state, row[dof], False)
        model.realizePosition(state)

        K_rec = {"time": row["time"]}
        for di, dname in enumerate(active_ik_dofs):
            K_rec[f"K_{dname}"] = K_mat[di, di]
        if n_dofs >= 2:
            K_rec["K_cross"] = K_mat[0, 1]
        K_rec["K_e_xx"]   = float(K_e[0, 0])
        K_rec["K_e_yy"]   = float(K_e[1, 1])
        K_rec["K_e_zz"]   = float(K_e[2, 2])
        K_rec["K_e_xy"]   = float(K_e[0, 1])
        K_rec["K_e_xz"]   = float(K_e[0, 2])
        K_rec["K_e_yz"]   = float(K_e[1, 2])
        K_rec["lambda_min"] = lam_min
        K_rec["lambda_max"] = lam_max
        K_records.append(K_rec)

    return pd.DataFrame(fl_records), pd.DataFrame(act_records), pd.DataFrame(K_records)


def plot_cocontraction_stiffness(act_df:  pd.DataFrame,
                                 K_df:    pd.DataFrame,
                                 save_path: str):
    """
    4-panel plot:
      1. Muscle activations (elbow flexors + extensors)
      2. Co-contraction index  CCI = 2·min(A_flex, A_ext) / (A_flex + A_ext)
      3. Joint stiffness  K_elbow_flex and K_elv_angle  [N·m/rad]
      4. Endpoint stiffness diagonals K_e_xx, K_e_yy + λ_min, λ_max [N/m]
    """
    t    = act_df["time"].values
    cmap = matplotlib.colormaps["tab10"]

    flex_ms = [m for m in ["BIClong", "BICshort", "BRA", "BRD"]     if m in act_df]
    ext_ms  = [m for m in ["TRIlong", "TRIlat", "TRImed"]           if m in act_df]
    shl_ms  = [m for m in ["DELT1", "DELT2", "DELT3"]               if m in act_df]

    A_flex = act_df[flex_ms].sum(axis=1).values
    A_ext  = act_df[ext_ms].sum(axis=1).values
    CCI    = 2.0 * np.minimum(A_flex, A_ext) / (A_flex + A_ext + 1e-9)

    has_Ke = "K_e_xx" in K_df.columns
    n_panels = 4 if has_Ke else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.0 * n_panels), sharex=True)

    # ── Panel 1: activations ──
    ax1 = axes[0]
    for idx, ms in enumerate(flex_ms):
        ax1.plot(t, act_df[ms].values, color=cmap(idx),
                 linewidth=1.5, label=f"{ms} (flex)")
    for idx, ms in enumerate(ext_ms):
        ax1.plot(t, act_df[ms].values, color=cmap(idx + 4),
                 linewidth=1.5, linestyle="--", label=f"{ms} (ext)")
    ax1.axvline(T_PERT, color="red", linewidth=0.9, linestyle=":", alpha=0.8,
                label=f"Perturbation t={T_PERT}s")
    ax1.set_ylabel("Activation (SO)", fontsize=9)
    ax1.set_title("Elbow muscle activations from Static Optimisation",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=7, ncol=4, loc="upper right")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.25)

    # ── Panel 2: CCI ──
    ax2 = axes[1]
    ax2.fill_between(t, CCI, alpha=0.35, color="#E91E63")
    ax2.plot(t, CCI, color="#E91E63", linewidth=2.0, label="CCI elbow")
    ax2.axvline(T_PERT, color="red", linewidth=0.9, linestyle=":", alpha=0.8)
    ax2.set_ylabel("CCI  (0–1)", fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Co-contraction index  CCI = 2·min(A_flex, A_ext) / (A_flex + A_ext)",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)

    # ── Panel 3: joint stiffness ──
    ax3 = axes[2]
    for col, clr, lbl in [
        ("K_elbow_flexion", "#2196F3", "K_elbow_flex  (N·m/rad)"),
        ("K_elv_angle",     "#FF5722", "K_elv_angle   (N·m/rad)"),
    ]:
        if col in K_df.columns:
            ax3.plot(K_df["time"].values, K_df[col].values,
                     color=clr, linewidth=2.0, label=lbl)
    ax3.axvline(T_PERT, color="red", linewidth=0.9, linestyle=":", alpha=0.8)
    ax3.set_ylabel("Joint stiffness (N·m/rad)", fontsize=9)
    ax3.set_xlabel("Time (s)", fontsize=10)
    ax3.set_title("SO-derived joint stiffness  K = R · diag(β·a·F₀·fl/ℓ_opt) · Rᵀ",
                  fontsize=10, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.25)

    # ── Panel 4: endpoint stiffness  K_e = J⁺ᵀ K_joint J⁺  ──
    if has_Ke:
        ax4 = axes[3]
        tk = K_df["time"].values
        ax4.plot(tk, K_df["K_e_xx"].values, color="#2196F3", linewidth=1.6,
                 label="K_e_xx  (N/m)")
        ax4.plot(tk, K_df["K_e_yy"].values, color="#FF5722", linewidth=1.6,
                 label="K_e_yy  (N/m)")
        ax4.plot(tk, K_df["lambda_min"].values, color="#4CAF50", linewidth=2.0,
                 linestyle="--", label="λ_min")
        ax4.plot(tk, K_df["lambda_max"].values, color="#9C27B0", linewidth=2.0,
                 linestyle="--", label="λ_max")
        ax4.axvline(T_PERT, color="red", linewidth=0.9, linestyle=":", alpha=0.8)
        ax4.set_ylabel("Endpoint stiffness (N/m)", fontsize=9)
        ax4.set_xlabel("Time (s)", fontsize=10)
        ax4.set_title("Endpoint stiffness  K_e = J⁺ᵀ · K_joint · J⁺   "
                      "(λ = principal axes of XY ellipse)",
                      fontsize=10, fontweight="bold")
        ax4.legend(fontsize=8, ncol=2)
        ax4.grid(True, alpha=0.25)

    fig.suptitle("Co-contraction, Joint and Endpoint Stiffness from Static Optimisation",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_active_vs_passive(ms_passive: pd.DataFrame,
                           ms_active:  pd.DataFrame,
                           save_path:  str):
    """
    4-panel comparison plot:
      dashed = passive (kinematic only)
      solid  = active  (SO-corrected, includes F_int effect)
    """
    t = ms_passive["time"].values
    GROUPS = {
        "Elbow flexors":   ["BIClong", "BICshort", "BRA", "BRD"],
        "Elbow extensors": ["TRIlong", "TRIlat", "TRImed"],
        "Shoulder":        ["DELT1", "DELT2", "DELT3"],
        "Wrist":           ["ECRL", "ECRB", "FCR", "FCU"],
    }
    cmap = matplotlib.colormaps["tab10"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.flatten()

    for ax, (group_name, muscles) in zip(axes, GROUPS.items()):
        for idx, ms in enumerate(muscles):
            if ms not in ms_passive.columns:
                continue
            col = cmap(idx)
            ax.plot(t, ms_passive[ms].values, color=col, linewidth=1.2,
                    linestyle="--", alpha=0.6, label=f"{ms} passive")
            ax.plot(t, ms_active[ms].values,  color=col, linewidth=2.0,
                    linestyle="-",             label=f"{ms} active (SO)")
        ax.axvline(T_PERT, color="red", linewidth=0.9, linestyle=":", alpha=0.7)
        ax.set_title(group_name, fontsize=10, fontweight="bold")
        ax.set_ylabel("Norm. fiber length (ℓ/ℓ₀)", fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.legend(fontsize=6, ncol=2, loc="best")
        ax.grid(True, alpha=0.25)

    fig.suptitle("Active (SO) vs Passive fiber lengths — effect of F_int on muscle state",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ===========================================================================
# Main
# ===========================================================================

# ===========================================================================
# Class wrapper — ArmInverseDynamics
# ===========================================================================
class ArmInverseDynamics:
    """Object-oriented driver for the perturbed-trajectory ID + active-MSK pipeline.

    Wraps the full ID → loop-closure → SO + joint-stiffness flow that previously
    lived in __main__. Module-level helpers remain importable for back-compat.
    """

    def __init__(self,
                 tag: str = "perturb",
                 perturb_dir: str = PERTURB_DIR,
                 output_dir:  str = OUTPUT_DIR,
                 model_path:  str = MODEL_PATH,
                 ms_labels:   list = None,
                 active_ik_dofs: list = None):
        self.tag         = tag
        self.perturb_dir = perturb_dir
        self.output_dir  = output_dir
        self.model_path  = model_path
        self.ms_labels   = list(ms_labels) if ms_labels is not None else list(MS_LABELS)
        self.active_ik_dofs = list(active_ik_dofs) if active_ik_dofs is not None \
            else list(ACTIVE_IK_DOFS)
        os.makedirs(self.output_dir, exist_ok=True)

    # ── End-to-end ────────────────────────────────────────────────────────
    def run(self) -> dict:
        TAG = self.tag

        # ── 1. Load trajectory + signals ──────────────────────────────────
        traj_csv = os.path.join(self.perturb_dir, f"cup_task_trajectory_{TAG}.csv")
        sig_csv  = os.path.join(self.perturb_dir, f"cup_task_signals_{TAG}.csv")

        traj  = pd.read_csv(traj_csv)
        sigs  = pd.read_csv(sig_csv)
        t     = traj["time"].values
        q_ea  = traj[["elv_angle", "elbow_flexion"]].values
        F_int = sigs["F_int"].values

        dt = t[1] - t[0]
        print(f"Loaded {len(t)} frames   Δt = {dt*1e3:.1f} ms   T = {t[-1]:.3f} s")

        # ── 2. CubicSpline derivatives → qdot, qddot ──────────────────────
        cs    = [CubicSpline(t, q_ea[:, j]) for j in range(2)]
        qdot  = np.stack([cs[j](t, 1) for j in range(2)], axis=1)
        qddot = np.stack([cs[j](t, 2) for j in range(2)], axis=1)

        print(f"\n  DOF               qdot_max (rad/s)    qddot_max (rad/s²)")
        for j, dof in enumerate(self.active_ik_dofs):
            print(f"  {dof:20s}   {np.abs(qdot[:,j]).max():.4f}             "
                  f"{np.abs(qddot[:,j]).max():.4f}")

        # ── 3. FK + Jacobians ─────────────────────────────────────────────
        print("\nLoading MoBL-ARMS 4.1 for FK + Jacobian…")
        model, state = _init_model()

        hand_pos = np.zeros((len(t), 3))
        J_all    = np.zeros((len(t), 3, 2))
        for i in tqdm(range(len(t)), desc="FK + J", ncols=80):
            hand_pos[i] = _hand_pos(model, state, q_ea[i])
            J_all[i]    = compute_jacobian(model, state, q_ea[i])

        d_raw     = hand_pos[-1] - hand_pos[0]
        direction = d_raw / np.linalg.norm(d_raw)
        print(f"  Motion direction : {np.round(direction, 3)}")

        # ── 4. External wrench on hand ────────────────────────────────────
        F_on_hand = -F_int[:, None] * direction[None, :]

        # ── 5. Write external-force files ─────────────────────────────────
        mot_path    = os.path.join(self.perturb_dir, f"cup_task_trajectory_{TAG}.mot")
        ext_mot     = os.path.join(self.output_dir, f"cup_task_ext_forces_{TAG}.mot")
        ext_xml     = os.path.join(self.output_dir, f"cup_task_ext_loads_{TAG}.xml")
        id_setup    = os.path.join(self.output_dir, f"cup_task_id_setup_{TAG}.xml")
        id_sto_name = f"cup_task_id_{TAG}.sto"
        id_sto      = os.path.join(self.output_dir, id_sto_name)

        write_ext_forces_mot(ext_mot, t, F_on_hand, hand_pos)
        write_ext_loads_xml(ext_xml, ext_mot)
        write_id_setup_xml(
            id_setup, results_dir=self.output_dir, model_file=self.model_path,
            mot_path=mot_path, sto_basename=id_sto_name,
            t_start=float(t[0]), t_end=float(t[-1]), ext_loads_xml=ext_xml,
        )
        id_setup_base    = os.path.join(self.output_dir, f"cup_task_id_setup_{TAG}_baseline.xml")
        id_sto_base_name = f"cup_task_id_{TAG}_baseline.sto"
        id_sto_base      = os.path.join(self.output_dir, id_sto_base_name)
        write_id_setup_xml(
            id_setup_base, results_dir=self.output_dir, model_file=self.model_path,
            mot_path=mot_path, sto_basename=id_sto_base_name,
            t_start=float(t[0]), t_end=float(t[-1]),
        )
        print(f"\nWrote: {ext_mot}\nWrote: {ext_xml}\nWrote: {id_setup}")

        # ── 6. Run InverseDynamicsTool ────────────────────────────────────
        print("\nRunning InverseDynamicsTool (with F_int)…  (~10–30 s)")
        osim.Logger.setLevel(osim.Logger.Level_Error)
        if not osim.InverseDynamicsTool(id_setup).run():
            raise RuntimeError("InverseDynamicsTool.run() returned False")
        print(f"ID complete → {id_sto}")

        print("Running InverseDynamicsTool (baseline, no ext force)…")
        ok_base = osim.InverseDynamicsTool(id_setup_base).run()
        osim.Logger.setLevel(osim.Logger.Level_Info)
        if not ok_base:
            raise RuntimeError("Baseline InverseDynamicsTool.run() returned False")
        print(f"Baseline ID complete → {id_sto_base}")

        # ── 7. Parse joint torques ────────────────────────────────────────
        tau_df  = read_sto(id_sto)
        print(f"\nID output columns: {list(tau_df.columns)}")
        t_id   = tau_df["time"].values

        col_elv  = _find_col(tau_df, "elv_angle")
        col_flex = _find_col(tau_df, "elbow_flexion")
        tau_elv  = tau_df[col_elv].values  if col_elv  else np.zeros(len(t_id))
        tau_flex = tau_df[col_flex].values if col_flex else np.zeros(len(t_id))
        if col_elv  is None: print("  WARNING: elv_angle torque not found in ID output")
        if col_flex is None: print("  WARNING: elbow_flexion torque not found in ID output")

        tau_arr = np.stack([tau_elv, tau_flex], axis=1)
        print(f"\n  τ_elv_angle   : [{tau_elv.min():.2f},  {tau_elv.max():.2f}] N·m")
        print(f"  τ_elbow_flex  : [{tau_flex.min():.2f}, {tau_flex.max():.2f}] N·m")
        print(f"  Peak torque at perturbation? "
              f"{tau_arr[np.argmin(np.abs(t_id - T_PERT)), :].round(2)} N·m")

        tau0_df  = read_sto(id_sto_base)
        t_id0    = tau0_df["time"].values
        col0_elv  = _find_col(tau0_df, "elv_angle")
        col0_flex = _find_col(tau0_df, "elbow_flexion")
        tau0_elv  = tau0_df[col0_elv].values  if col0_elv  else np.zeros(len(t_id0))
        tau0_flex = tau0_df[col0_flex].values if col0_flex else np.zeros(len(t_id0))
        if len(t_id0) != len(t_id):
            tau0_elv  = interp1d(t_id0, tau0_elv,  bounds_error=False,
                                 fill_value="extrapolate")(t_id)
            tau0_flex = interp1d(t_id0, tau0_flex, bounds_error=False,
                                 fill_value="extrapolate")(t_id)
        tau0_arr = np.stack([tau0_elv, tau0_flex], axis=1)
        print(f"  τ₀_elv_angle  (baseline): [{tau0_elv.min():.2f}, {tau0_elv.max():.2f}] N·m")
        print(f"  τ₀_elbow_flex (baseline): [{tau0_flex.min():.2f}, {tau0_flex.max():.2f}] N·m")

        # ── 8. Loop-closure ───────────────────────────────────────────────
        if len(t_id) != len(t):
            J_id = np.zeros((len(t_id), 3, 2))
            for r in range(3):
                for c in range(2):
                    J_id[:, r, c] = interp1d(t, J_all[:, r, c],
                                              bounds_error=False,
                                              fill_value="extrapolate")(t_id)
        else:
            J_id = J_all

        j_line          = J_id.transpose(0, 2, 1) @ direction
        delta_tau       = tau_arr - tau0_arr
        numer           = np.einsum('ij,ij->i', j_line, delta_tau)
        denom           = np.einsum('ij,ij->i', j_line, j_line)
        F_int_recovered = numer / (denom + 1e-10)

        print(f"\n  F_int_recovered range : [{F_int_recovered.min():.2f}, {F_int_recovered.max():.2f}] N")
        print(f"  F_int prescribed range: [{F_int.min():.2f}, {F_int.max():.2f}] N")

        # ── 9. Save results CSV ───────────────────────────────────────────
        res_csv = os.path.join(self.output_dir, f"cup_task_id_results_{TAG}.csv")
        pd.DataFrame({
            "time":              t_id,
            "tau_elv_angle":     tau_elv,
            "tau_elbow_flex":    tau_flex,
            "tau0_elv_angle":    tau0_elv,
            "tau0_elbow_flex":   tau0_flex,
            "delta_tau_elv":     delta_tau[:, 0],
            "delta_tau_flex":    delta_tau[:, 1],
            "F_int_recovered":   F_int_recovered,
        }).to_csv(res_csv, index=False)
        print(f"\nSaved: {res_csv}")

        F_int_id = (interp1d(t, F_int, bounds_error=False, fill_value=0.0)(t_id)
                    if len(t_id) != len(t) else F_int)

        # ── 10. Plots ─────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        dof_labels = ["elv_angle  (N·m)", "elbow_flexion  (N·m)"]
        colors     = ["#2196F3", "#FF5722"]
        for j, (ax, lbl, clr) in enumerate(zip(axes, dof_labels, colors)):
            ax.plot(t_id, tau_arr[:, j], color=clr, linewidth=2, label=f"τ  {lbl}")
            ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
            ax.axvline(T_PERT, color="red", linewidth=0.9, linestyle=":", alpha=0.8,
                       label=f"Perturbation  t = {T_PERT} s")
            ax.set_ylabel(f"τ  {lbl}", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)", fontsize=11)
        axes[0].set_title("Joint Torques — Inverse Dynamics with F_int external wrench", fontsize=11)
        fig.tight_layout()
        p1 = os.path.join(self.output_dir, f"cup_task_id_torques_{TAG}.png")
        fig.savefig(p1, dpi=150); plt.close(fig)
        print(f"Saved: {p1}")

        fig, axes2 = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax2a = axes2[0]
        ax2a.plot(t_id, delta_tau[:, 0], color="#2196F3", linewidth=2,
                  label="Δτ  elv_angle  (N·m)")
        ax2a.plot(t_id, delta_tau[:, 1], color="#FF5722", linewidth=2,
                  label="Δτ  elbow_flexion  (N·m)")
        ax2a.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax2a.axvline(T_PERT, color="red", linewidth=0.9, linestyle=":", alpha=0.8)
        ax2a.set_ylabel("Δτ = τ − τ₀  (N·m)", fontsize=10)
        ax2a.set_title("Loop-closure check: Δτ (ext-force signature) and F_int recovery",
                       fontsize=11)
        ax2a.legend(fontsize=8, loc="upper right"); ax2a.grid(True, alpha=0.3)

        ax2b = axes2[1]
        ax2b.plot(t_id, F_int_id, color="#9B30FF", linewidth=2,
                  label="F_int  prescribed (N)")
        ax2b.plot(t_id, F_int_recovered, color="#FF9800", linewidth=2, linestyle="--",
                  label="F_int  recovered  (j_line · Δτ / |j_line|²)  (N)")
        ax2b.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax2b.axvline(T_PERT, color="red", linewidth=0.9, linestyle=":", alpha=0.8)
        ax2b.set_xlabel("Time (s)", fontsize=11)
        ax2b.set_ylabel("Force (N)", fontsize=11)
        ax2b.legend(fontsize=8, loc="upper right"); ax2b.grid(True, alpha=0.3)
        fig.tight_layout()
        p2 = os.path.join(self.output_dir, f"cup_task_id_loopclosure_{TAG}.png")
        fig.savefig(p2, dpi=150); plt.close(fig)
        print(f"Saved: {p2}")

        # ── 11. Active (SO-corrected) MSK + joint stiffness ───────────────
        print("\nComputing activation-corrected fiber lengths (Static Optimisation)…")
        passive_csv = os.path.join(self.perturb_dir, f"cup_task_fiber_lengths_{TAG}.csv")
        ms_passive  = pd.read_csv(passive_csv)
        tau_on_traj = np.stack([
            interp1d(t_id, tau_elv,  bounds_error=False, fill_value="extrapolate")(t),
            interp1d(t_id, tau_flex, bounds_error=False, fill_value="extrapolate")(t),
        ], axis=1)

        ms_active, act_df, K_df = run_active_msk(
            traj, tau_on_traj, self.model_path, self.ms_labels, self.active_ik_dofs)

        active_csv = os.path.join(self.output_dir, f"cup_task_active_fiber_lengths_{TAG}.csv")
        act_csv    = os.path.join(self.output_dir, f"cup_task_activations_{TAG}.csv")
        K_csv      = os.path.join(self.output_dir, f"cup_task_joint_stiffness_{TAG}.csv")

        ms_active.to_csv(active_csv, index=False)
        act_df.to_csv(act_csv,    index=False)
        K_df.to_csv(K_csv,        index=False)
        print(f"Saved: {active_csv}\nSaved: {act_csv}\nSaved: {K_csv}")

        plot_active_vs_passive(ms_passive, ms_active,
            os.path.join(self.output_dir, f"cup_task_active_vs_passive_fl_{TAG}.png"))
        plot_cocontraction_stiffness(act_df, K_df,
            os.path.join(self.output_dir, f"cup_task_cocontraction_stiffness_{TAG}.png"))

        print("\nDone.")
        print("  Next: python arm_inverse_dynamics.py --tag linear   (unperturbed)")
        return {"id_results_csv": res_csv,
                "active_csv":     active_csv,
                "act_csv":        act_csv,
                "K_csv":          K_csv}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Inverse Dynamics with F_int external wrench on hand")
    ap.add_argument("--tag", default="perturb",
                    help="Trajectory tag to load/save (default: perturb)")
    args = ap.parse_args()
    ArmInverseDynamics(tag=args.tag).run()
