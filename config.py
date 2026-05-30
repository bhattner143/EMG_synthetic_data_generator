"""
config.py — single source of truth for simulation globals.

All constants used across the cup-task pipeline (arm_cup_task,
arm_cup_perturbation, arm_inverse_dynamics, compute_stiffness,
compare_to_razavian2021) live here.

Two access patterns are supported:

  1. Flat module-level constants (backward-compatible):
       from config import FS, MS_LABELS, ACTIVE_DOFS, NEUTRAL, MODEL_PATH

  2. Dataclass instance (preferred for new code, avoids implicit globals):
       from config import CFG
       fs = CFG.kinematics.fs

The flat constants are kept in lockstep with the dataclass defaults — change
one, the other follows.

Run this file directly (`python config.py`) to dump the full configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Tuple

import numpy as np


# ===========================================================================
# Paths
# ===========================================================================
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(SCRIPT_DIR, "model", "MOBL_ARMS_41.osim")
DEMO_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "demo_output")


# ===========================================================================
# Kinematics
# ===========================================================================
FS = 100   # Hz – kinematic sample rate

# 7 active DOFs of MoBL-ARMS 4.1 driven by the cup-task pipeline
ACTIVE_DOFS = [
    "elv_angle",
    "shoulder_elv",
    "shoulder_rot",
    "elbow_flexion",
    "pro_sup",
    "deviation",
    "flexion",
]

# Reduced 2-DOF set for IK / ID in the perturbation pipeline
ACTIVE_IK_DOFS = ["elv_angle", "elbow_flexion"]

# Neutral / reference hold posture (radians)
NEUTRAL = {
    "elv_angle":     np.radians(52.857),
    "shoulder_elv":  np.radians(80.429),
    "shoulder_rot":  np.radians(58.0),
    "elbow_flexion": np.radians(65.619),
    "pro_sup":       np.radians(-0.857),
    "deviation":     np.radians(0.069),
    "flexion":       np.radians(-0.338),
}


# ===========================================================================
# Muscles (surface-EMG-relevant)
# ===========================================================================
MS_LABELS = [
    "BIClong", "BICshort",
    "TRIlong", "TRIlat", "TRImed",
    "BRA", "BRD",
    "DELT1", "DELT2", "DELT3",
    "ECRL", "ECRB",
    "FCR",  "FCU",
]

HAND_BODY     = "hand"
HAND_LOCAL_PT = (0.0, 0.0, 0.0)


# ===========================================================================
# Static optimisation / Hill model
# ===========================================================================
# Tonic baseline activation floor (~2 % MVC) — keeps EMG channels non-silent
BASELINE_ALPHA      = 0.02

# Co-contraction floor injected during perturbation windows (CMC-style)
COCONTRACTION_ALPHA = 0.15

# Gaussian force–length curve width (Hill, γ ≈ 0.45)
FL_GAMMA            = 0.45

# Stiffness scale in compute_stiffness (β=1 ⇒ k_m = a·F0·fl/l_opt)
BETA_STIFFNESS_K_E       = 1.0
# Stiffness scale in arm_inverse_dynamics (β=40 used historically there)
BETA_STIFFNESS_JOINT_K   = 40.0

# Hill force–velocity damping linearisation around v_m = 0
#   d_m = BETA_D · a · F0 / (V_MAX_FACTOR · l_opt)
BETA_D       = 0.3
V_MAX_FACTOR = 10.0  # 1/s


# ===========================================================================
# Perturbation trajectory parameters (arm_cup_perturbation.py)
# ===========================================================================
T_MOVE     = 1.60     # s
D_MOVE     = 0.30     # m
M_EFF      = 0.1      # kg (effective endpoint inertia)
T_PERT     = 0.90     # s
F_PERT     = 12.0     # N
SIGMA_PERT = 0.030    # s
V_DIP_FRAC = 0.85
F_NEG      = 0.5      # N
DT_NEG     = 0.20     # s
SIG_NEG    = 0.15     # s

ELV_ANGLE_BOUNDS  = (np.radians(-30), np.radians(120))
ELBOW_FLEX_BOUNDS = (np.radians(25),  np.radians(160))


# ===========================================================================
# Razavian et al. 2021 (ICRA) reference values
# ===========================================================================
KP_PAPER       = 40.0   # N/m
KD_PAPER       = 50.0   # N·s/m
FPERT_PAPER    = -20.0  # N
PERT_DUR_PAPER = 0.020  # s


# ===========================================================================
# Dataclass aggregator (preferred surface for new code)
# ===========================================================================
@dataclass(frozen=True)
class Paths:
    script_dir:  str = SCRIPT_DIR
    model_path:  str = MODEL_PATH
    output_dir:  str = DEMO_OUTPUT_DIR


@dataclass(frozen=True)
class Kinematics:
    fs:             int        = FS
    active_dofs:    Tuple[str, ...] = tuple(ACTIVE_DOFS)
    active_ik_dofs: Tuple[str, ...] = tuple(ACTIVE_IK_DOFS)
    neutral:        dict       = field(default_factory=lambda: dict(NEUTRAL))


@dataclass(frozen=True)
class Muscles:
    labels:        Tuple[str, ...]        = tuple(MS_LABELS)
    hand_body:     str                    = HAND_BODY
    hand_local_pt: Tuple[float, float, float] = HAND_LOCAL_PT


@dataclass(frozen=True)
class Hill:
    baseline_alpha:        float = BASELINE_ALPHA
    cocontraction_alpha:   float = COCONTRACTION_ALPHA
    fl_gamma:              float = FL_GAMMA
    beta_stiffness_k_e:    float = BETA_STIFFNESS_K_E
    beta_stiffness_joint:  float = BETA_STIFFNESS_JOINT_K
    beta_d:                float = BETA_D
    v_max_factor:          float = V_MAX_FACTOR


@dataclass(frozen=True)
class Perturbation:
    t_move:     float = T_MOVE
    d_move:     float = D_MOVE
    m_eff:      float = M_EFF
    t_pert:     float = T_PERT
    f_pert:     float = F_PERT
    sigma_pert: float = SIGMA_PERT
    v_dip_frac: float = V_DIP_FRAC
    f_neg:      float = F_NEG
    dt_neg:     float = DT_NEG
    sig_neg:    float = SIG_NEG
    elv_angle_bounds:  Tuple[float, float] = ELV_ANGLE_BOUNDS
    elbow_flex_bounds: Tuple[float, float] = ELBOW_FLEX_BOUNDS


@dataclass(frozen=True)
class Razavian2021:
    kp:       float = KP_PAPER
    kd:       float = KD_PAPER
    f_pert:   float = FPERT_PAPER
    pert_dur: float = PERT_DUR_PAPER


@dataclass(frozen=True)
class SimConfig:
    paths:        Paths        = field(default_factory=Paths)
    kinematics:   Kinematics   = field(default_factory=Kinematics)
    muscles:      Muscles      = field(default_factory=Muscles)
    hill:         Hill         = field(default_factory=Hill)
    perturb:      Perturbation = field(default_factory=Perturbation)
    razavian2021: Razavian2021 = field(default_factory=Razavian2021)


CFG = SimConfig()


# ===========================================================================
# Self-test / dump
# ===========================================================================
def dump(cfg: SimConfig = CFG) -> None:
    import json
    print(json.dumps(asdict(cfg), indent=2, default=str))


if __name__ == "__main__":
    dump()
