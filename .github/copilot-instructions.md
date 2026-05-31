# Project: Ball-in-Cup Experimental Platform — EMG Synthetic Data Generator

## Application and Clinical Motivation
This codebase supports the development of an **adaptive elbow exosuit for people with Parkinson's
disease (PD) and related neurological movement disorders**. PD causes rigidity, tremor, and
bradykinesia in the upper limb, degrading the patient's ability to modulate arm stiffness
voluntarily. The exosuit compensates by:
1. **Observing** healthy-subject EMG and stiffness patterns during a functionally relevant task
   (ball-in-cup manipulation).
2. **Replicating** those patterns on a cable-driven elbow exosuit via a learned Tele-Impedance
   policy.
3. **Augmenting** the patient's residual muscle activity with appropriately timed cable pre-tension
   so the arm behaves as if stiffness modulation were intact.

The synthetic EMG pipeline enables **data augmentation and policy pre-training** without requiring
large cohorts of real patients — critical for a neurological population where recruitment is slow
and fatigue is limiting.

---

## Overarching Goal
Develop a human-inspired stiffness modulation framework for elbow exosuits. Healthy human
demonstrations on a two-rail ball-in-cup platform provide the target impedance behaviour. A
cable-driven manipulator with an antagonistic exosuit replicates and learns that behaviour through
a three-phase pipeline.

---

## End-to-End Computational Pipeline

```
[Human experiment]          [OpenSim musculoskeletal model]        [Neuromuscular synthesis]
  2-axis rail (XY)   ──►  arm_cup_perturbation.py  ──►  q(t)
  8-ch surface EMG         (min-jerk + Gaussian pert,              ┌─ compute_stiffness.py
  6-DOF F/T sensor          numerical IK, F_int signal)            │    SO activations a_m(t)
  Optitrack pos             │                                       │    K_e(t), D_e(t), p_null
                            ▼                                       │
                       arm_inverse_dynamics.py                      │    generate_emg_upper_limb.py
                          (7-DOF ID + CMC static opt)               │      ├── Fuglevand MN pool
                            │                                       │      │   (spike trains per MU)
                            ▼                                       │      └── BioMime VAE
                       compute_stiffness.py ──────────────────────►│          (MUAP grid 10×32)
                          (Hill stiffness chain,                    │             │
                           endpoint K_e, D_e,                       │             ▼
                           co-contraction p_null)                   │    8-ch synthetic sEMG
                            │                                       │    (matches real electrode bank)
                            ▼                                       │
                    compare_to_razavian2021.py                      │
                       (validates K_e vs 40 N/m,                    │
                        D_e vs 50 N·s/m benchmarks)                 │
                                                                    ▼
[Robot / simulator]                                         [RL / imitation learning]
  PyDrake / Isaac Sim                                         p_null → c_exo (SEA cable setpoint)
  CT controller                                               K_e(t) → reward shaping
  SEA exo cables                                              synthetic sEMG → policy observation
  PPO residual policy
```

---

## Three-Phase Experiment Pipeline
- **Phase 1 — Human Rail Experiment (master side):** Participant moves a 2-axis rail (X = left/right,
  Y = up/down) while keeping a ball in a cup. Sensors: 8-channel surface EMG on the arm, 6-DOF
  F/T at the cup handle, Optitrack/rail encoders for position, overhead camera for ball tracking.
  Produces trajectory reference `(xc(t), yc(t))`, force reference `F(t)`, EMG envelope `P(t)`, and
  ball-state labels (dwell time, boundary violations, recovery time).
- **Phase 2 — Exo + Manipulator digital twin (PyDrake / Isaac Sim):** Rail position is fed into
  an IK solver to generate joint references `q*(t)`. A weak-stiffness CT controller tracks the
  trajectory. Human EMG null-space co-contraction `P_null` drives the SEA exo-cable setpoint
  `c_exo`. An RL/imitation policy (PPO) is trained on closed-loop rollout data.
- **Phase 3 — Real-robot validation:** The trained policy replaces the EMG-based Tele-Impedance
  signal; the CT controller handles position tracking while the learned policy drives exo cable
  pre-tension. Evaluated by comparing Exo OFF vs Exo ON conditions.

## Tele-Impedance Architecture (Ajoudani et al., 2012)
- **Master:** participant at the 2-rail platform with 8-ch EMG, F/T, Optitrack.
- **Slave:** 2-DOF cable-driven manipulator with SEA elbow exosuit.
- Two-stage calibration: (1) EMG → endpoint-force map `T̂` (isometric regression), (2) null-space
  EMG → stiffness map `Ψ̂` (perturbation-based second-order impedance identification).
- Data flow at ~200 Hz: rail position → IK → CT controller; EMG → null-space projector → Ψ̂ →
  `c_exo` → SEA exo cables.

---

## OpenSim Model for Cup Manipulation Task
**Goal:** Create an OpenSim model that represents the cup-manipulation task so that:
1. The MoBL-ARMS 4.1 upper limb model (`repo/MoBL-ARMS Upper Extremity Model/Model/4.1/MOBL_ARMS_41.osim`)
   drives the cup handle through the 2-D rail trajectories `(xc(t), yc(t))`.
2. Muscle fiber lengths, activation levels, and joint torques are extracted along these trajectories
   to provide EMG-relevant ground truth for synthetic data generation.
3. The ball-in-cup dynamics can optionally be represented as a pendulum-like body attached to the
   cup frame (underactuated, free to swing in XY).

### Relevant model components
- **Base model:** `MOBL_ARMS_41.osim` — 7 active DOFs (elv_angle, shoulder_elv, shoulder_rot,
  elbow_flexion, pro_sup, deviation, flexion) plus many locked phantom DOFs.
- **Neutral posture:** `elbow_flexion=π/2 (90°)`, `shoulder_elv=π/6 (30°)`, others = 0.
- **Cup task muscles (surface-EMG relevant):**
  - Elbow flexors: BIClong, BICshort, BRA, BRD
  - Elbow extensors: TRIlong, TRIlat, TRImed
  - Shoulder: DELT1, DELT2, DELT3
  - Wrist/forearm: ECRL, ECRB, FCR, FCU
- **Trajectory driver:** `arm_cup_task.py` — drives the model through sine or min-jerk rail
  trajectories and saves fiber lengths to `demo_output/`.

---

## Key Files and What Each Script Does

| File | Class | What it does |
|---|---|---|
| `config.py` | `SimConfig` (frozen dataclass) | **Single source of truth** — FS=100 Hz, MS_LABELS, ACTIVE_DOFS, NEUTRAL pose, MODEL_PATH, Hill-muscle params (BASELINE_ALPHA, FL_GAMMA, BETA_D), perturbation params, Razavian 2021 reference values. Import via `from config import CFG`. |
| `arm_cup_task.py` | `ArmCupTask` | Drives MoBL-ARMS 4.1 through sine (Lissajous) or min-jerk rail trajectories using the OpenSim Python API. Extracts muscle fiber lengths and writes `.mot` + CSV to `demo_output/arm_cup_task/`. |
| `arm_cup_perturbation.py` | `ArmCupPerturbation` | Generates a min-jerk XY cup trajectory with a superimposed Gaussian force perturbation (`--t-pert`, `--f-pert`, `--sigma`). Runs numerical IK (central-difference Jacobian, Newton-Raphson) to obtain `q(t)` for all 7 DOFs. Outputs trajectory CSV + fiber-length CSV + plots. |
| `arm_inverse_dynamics.py` | `ArmInverseDynamics` | Runs OpenSim `InverseDynamicsTool` with the external cup–hand wrench (`F_int`) to get joint torques `τ(t)`. Then runs static optimisation (Crowninshield–Brand cubic cost) to distribute torques across the 14-muscle set → per-muscle activations `a_m(t)` and joint stiffness `K_joint(t)`. |
| `compute_stiffness.py` | `ComputeStiffness` | Full 7-DOF endpoint stiffness pipeline. For each frame: builds muscle geometry (Jacobian, moment arms, pennation, fiber length ratio), applies Hill stiffness model `k_m = γ F_m / l_m`, assembles the joint stiffness matrix, maps to endpoint via `K_e = J^{-T} K_joint J^{-1}`. Also computes damping tensor `D_e(t)` and null-space co-contraction signal `p_null(t)` (CMC floor). Writes `cup_task_stiffness_perturb_cmc.csv` with columns `time, perturb, p_null, a_<muscle>..., K_e_xx..., D_e_xx..., K_min, K_max`. |
| `compare_to_razavian2021.py` | `RazavianComparison` | Quantitative validation: loads the stiffness CSV and compares `K_e_xx` vs k_p=40 N/m and `D_e_xx` vs k_d=50 N·s/m from Razavian et al. (ICRA 2021). Prints RMSE and peak/baseline ratios; saves comparison PNG and summary text. |
| `generate_emg_upper_limb.py` | *(script, no class)* | **Synthetic sEMG bridge.** Reads SO activations from `compute_stiffness.py` output (100 Hz), resamples to 2048 Hz, then for each of 14 upper-limb muscles: (1) runs a Fuglevand-style `MotoneuronPool` (NeuroMotion) to generate discrete MU spike trains from `a_m(t)`, (2) calls the pre-trained BioMime conditional VAE once to sample a 10×32-electrode MUAP grid per MU conditioned on anatomy parameters (depth, angle, fibre count, innervation zone, conduction velocity, fibre length), (3) convolves each MU's MUAP with its spike train and sums across all MUs. Final outputs: 8-channel sEMG bank (matching the real experiment's electrode placement), full 10×32 EMG grid, activation envelope plots. Saved to `demo_output/generate_emg_upper_limb/`. |
| `NeuroMotion/NeuroMotion/MNPoollib/mn_params.py` | *(params module)* | Anatomical parameter tables (DEPTH, ANGLE, MS_AREA, NUM_MUS) for all muscles. Extended with 11 upper-limb muscles: BIClong, BICshort, BRA, BRD, TRIlong/lat/med, DELT1/2/3, FCR. Areas derived from MoBL-ARMS `F_max / σ₀` (σ₀=0.61 N/mm²); MU counts from Enoka & Fuglevand 2001. |
| `NeuroMotion/ckp/model_linear.pth` | *(BioMime weights)* | Pre-trained BioMime VAE generator checkpoint (forearm training data). Used by `generate_emg_upper_limb.py` — morphology is extrapolated for upper arm but conditioning on anatomy parameters remains structurally valid. |
| `demo_output/<script_name>/` | — | Per-script output subdirectories (CSV, .mot, .png, .npz). |
| `repo/MoBL-ARMS Upper Extremity Model/Model/4.1/MOBL_ARMS_41.osim` | — | Base OpenSim upper-limb model (50 muscles, 12 bodies). |
| `repo/musculoskeletal-stiffness/arm_model/` | — | Feasible stiffness analysis (Stanev & Moustakas 2019). |
| `notes/pipeline_notes.tex` | — | Comprehensive LaTeX documentation (32 pages) of every pipeline stage, equations, and implementation decisions. |

---

## Execution Sequence (full pipeline)

```bash
# Step 1 — generate perturbed cup-task trajectory + fiber lengths
python arm_cup_perturbation.py

# Step 2 — inverse dynamics + static optimisation → joint torques + activations
python arm_inverse_dynamics.py

# Step 3 — endpoint stiffness K_e(t), damping D_e(t), co-contraction p_null(t)
python compute_stiffness.py --mode perturb --stiffness cmc

# Step 4 — validate against Razavian 2021 benchmarks
python compare_to_razavian2021.py

# Step 5 — synthetic 8-channel surface EMG (Mac/CPU, quick test)
python generate_emg_upper_limb.py --duration 1.0 --num_mus_cap 30 --device cpu
# Full run (Apple Silicon MPS, default ~200-350 MU/muscle)
python generate_emg_upper_limb.py --device mps
```

Downstream robot/RL pipeline:
- `p_null` column → `c_exo` (SEA exo cable pre-tension setpoint in `actuators/sea_isaacsim.py`)
- `K_e(t)` → reward shaping signal in `rl/train_ppo_residual.py`
- `emg_8ch` array → policy observation vector

---

## Synthetic EMG Methodology (`generate_emg_upper_limb.py`)

The script implements a two-stage neuromuscular model:

**Stage 1 — Fuglevand Motor Neuron Pool** (`NeuroMotion/MNPoollib/MNPool.py`)
- Motor units are assigned exponentially distributed recruitment thresholds and a linear
  firing-rate vs excitation-drive relationship (Fuglevand et al. 1993 model).
- The SO-derived activation `a_m(t)` is used directly as the excitation drive `ext(t)` (% MVC).
- Outputs: discrete spike trains per MU at 2048 Hz.

**Stage 2 — BioMime Conditional VAE** (`NeuroMotion/ckp/model_linear.pth`)
- A pre-trained generative model conditioned on 6 anatomy parameters per MU:
  `(num_fibres, depth, angle, iz, cv, fibre_length)`.
- Samples one MUAP shape: `generator.sample(N_MU, cond) → [N_MU, 10, 32, 96]`
  (10-row × 32-col HD-EMG grid, 96 samples at 2048 Hz ≈ 47 ms window).
- A single static draw per muscle is used (no MSK-driven length change); valid for
  envelope-level synthesis; individual MUAP morphologies are extrapolated for upper arm.

**Stage 3 — MUAP Convolution and Summation**
```
EMG(r, c, t) = Σ_muscles Σ_i Σ_{k ∈ spikes_i}  MUAP_i(r, c, t−k)
```
Surface EMG is then subsampled to 8 channels matching the real experimental electrode bank.

**Mac compatibility:** No CUDA required. Default device is `mps` (Apple Metal); falls back to
`cpu` automatically. BioMime timing on M-series Mac: N=50 MUs → ~2 s per muscle.

---

## Architecture (cup-task pipeline)
- **Class-based**: each script defines exactly one public class (`ArmCupTask`,
  `ArmCupPerturbation`, `ArmInverseDynamics`, `ComputeStiffness`,
  `RazavianComparison`). Calling `ClassName().run()` reproduces the legacy
  `__main__` behaviour. Steps can also be invoked individually for testing.
- **Backwards-compatible**: every legacy module-level function (`build_trajectory`,
  `run_msk`, `compute_moment_arms`, `so_frame`, etc.) remains importable; the
  classes are thin façades.
- **Centralised config**: prefer `from config import CFG` (or specific constants)
  over hard-coded numbers when adding new code.
- **Each script keeps `if __name__ == "__main__":`** so it can be run individually
  (e.g. `python compute_stiffness.py --mode perturb --stiffness cmc`).

## Simulation Entry Points (manipulator side)
- PyDrake: `script_cup_manipulator_pendulam_tendon_with_exo_pydrake.py`
- Isaac Sim: `script_cup_manipulator_pendulam_tendon_with_exo_isaac_sim.py`
- CT controller: `controller/computed_torque_isaacsim.py`
- SEA cable model: `actuators/sea_isaacsim.py`
- RL training: `rl/train_ppo_residual.py`

## Coding Conventions
- Python 3.12+, conda env `arm_emg` (`/opt/anaconda3/envs/arm_emg`).
- OpenSim via `import opensim as osim` — use `arm_emg` env which has OpenSim 4.4.
- Torch + BioMime installed in `arm_emg` env (torch 2.12, MPS-enabled for Mac).
- Matplotlib backend must be `'Agg'` when running headless.
- musculoskeletal-stiffness code requires `numpy <2.0` fixes already applied to `util.py`,
  `simulation.py`, and `analysis.py`; `lrs` binary needed for cases 1 and 5.
- Use `tqdm` for progress bars, `pandas` for CSV I/O.
- `generate_emg_upper_limb.py` uses no class; it is a standalone argparse script.

## Sample Size (human experiments)
Target N ≈ 12–20 healthy participants (within-subject design). Effect size from literature:
d_z ≈ 0.6–1.0. Power analysis: `statsmodels.stats.power.TTestPower` with α=0.05, power=0.80.
