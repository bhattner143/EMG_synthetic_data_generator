"""
generate_emg_upper_limb.py
==========================

Bridge from compute_stiffness.py muscle activations to multi-channel synthetic
surface EMG, for **upper-limb muscles** of the MoBL-ARMS 4.1 cup-task pipeline.

Pipeline (per muscle):
    a(t) at 100 Hz  --(resample)-->  ext(t) at 2048 Hz
        |                                 |
        |  (Fuglevand-style MNPool)       |  (single static draw, BioMime VAE)
        v                                 v
    spike_trains[mu]                  MUAP grid [num_mu, 10, 32, 96]
        \\____________________  ____________________/
                              \\/
              convolve --> per-muscle EMG[10, 32, T]
                              v
              sum across muscles --> sEMG[10, 32, T]
              subsample --> 8-channel sEMG matching the project's surface bank

Mac compatibility
-----------------
* No CUDA needed. Default device is `cpu`. Pass `--device mps` to use Apple Metal.
* Uses BioMime + NeuroMotion (already installed in conda env `arm_emg`).

Caveats
-------
BioMime's VAE was trained on forearm muscles. Conditioning on upper-arm anatomy
parameters (depth/angle/cv/length/iz/num_fibres) produces structurally valid
spike-driven envelopes, but individual MUAP morphologies are extrapolations and
should not be taken as ground-truth waveform shape for upper-arm-specific
biophysics studies.

Usage
-----
    python generate_emg_upper_limb.py
    python generate_emg_upper_limb.py --device mps --duration 6.0
    python generate_emg_upper_limb.py --csv demo_output/compute_stiffness/cup_task_stiffness_perturb_cmc.csv
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from scipy.signal import butter, filtfilt, resample_poly
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "NeuroMotion"))

from NeuroMotion.MNPoollib.MNPool import MotoneuronPool                                  # noqa: E402
from NeuroMotion.MNPoollib.mn_utils import generate_emg_mu                                # noqa: E402
from NeuroMotion.MNPoollib.mn_params import (                                             # noqa: E402
    DEPTH, ANGLE, MS_AREA, NUM_MUS, mn_default_settings,
)
from BioMime.models.generator import Generator                                            # noqa: E402
from BioMime.utils.basics import update_config, load_generator                            # noqa: E402


# Muscles in the compute_stiffness CSV that we'll synthesise EMG for.
UPPER_LIMB_MS = [
    "BIClong", "BICshort", "BRA", "BRD",
    "TRIlong", "TRIlat", "TRImed",
    "DELT1", "DELT2", "DELT3",
    "ECRL", "ECRB", "FCR", "FCU",
]

# 8-channel surface bank — pick rows of the 10x32 BioMime grid that roughly span
# the upper arm circumference. Each entry is (row_idx, col_idx) on the grid.
SURFACE_BANK_8CH = [
    (1, 8), (3, 8), (5, 8), (7, 8),
    (1, 24), (3, 24), (5, 24), (7, 24),
]


def load_activations(csv_path: Path, ms_labels):
    """Read compute_stiffness CSV and return (t, dict[label -> a(t)])."""
    df = pd.read_csv(csv_path)
    t = df["time"].to_numpy()
    acts = {}
    for ms in ms_labels:
        col = f"a_{ms}"
        if col not in df.columns:
            raise KeyError(f"Activation column {col!r} not found in {csv_path}")
        acts[ms] = df[col].to_numpy()
    return t, acts


def resample_to_fs(sig: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Polyphase resample with rational fs_out/fs_in. Returns float64."""
    from math import gcd
    fi, fo = int(round(fs_in)), int(round(fs_out))
    g = gcd(fi, fo)
    return resample_poly(sig, fo // g, fi // g).astype(np.float64)


def build_pool_for_muscle(ms_label: str, fs: int, fibre_density: int = 200,
                          num_mus_cap: int | None = None):
    """Instantiate MotoneuronPool, assign physiological properties, init twitches."""
    num_mus = NUM_MUS[ms_label]
    if num_mus_cap is not None:
        num_mus = min(num_mus, num_mus_cap)
    pool = MotoneuronPool(num_mus, ms_label, **mn_default_settings)
    num_fb = np.round(MS_AREA[ms_label] * fibre_density)
    cfg = edict({
        "num_fb": num_fb,
        "depth": DEPTH[ms_label],
        "angle": ANGLE[ms_label],
        "iz": [0.5, 0.1],
        "len": [1.0, 0.05],
        "cv": [4.0, 0.3],
    })
    properties = pool.assign_properties(cfg, normalise=True)
    pool.init_twitches(fs)
    pool.init_quisistatic_ef_model()
    return pool, properties, num_mus


def sample_static_muaps(generator, properties, num_mus, latent_dim, device, b, a):
    """Draw one set of MUAPs for the muscle (static muscle length assumption)."""
    num    = torch.from_numpy(properties["num"]).reshape(num_mus, 1).float()
    depth  = torch.from_numpy(properties["depth"]).reshape(num_mus, 1).float()
    angle  = torch.from_numpy(properties["angle"]).reshape(num_mus, 1).float()
    iz     = torch.from_numpy(properties["iz"]).reshape(num_mus, 1).float()
    cv     = torch.from_numpy(properties["cv"]).reshape(num_mus, 1).float()
    length = torch.from_numpy(properties["len"]).reshape(num_mus, 1).float()
    cond = torch.cat([num, depth, angle, iz, cv, length], dim=1).to(device)

    zi = torch.randn(num_mus, latent_dim, device=device)
    with torch.no_grad():
        sim = generator.sample(num_mus, cond, device, zi)   # [num_mus, 10, 32, 96]
    sim = sim.permute(0, 2, 3, 1).detach().cpu().numpy()    # [num_mus, 10, 32, 96]
    n_mu, n_row, n_col, n_t = sim.shape
    sim = filtfilt(b, a, sim.reshape(-1, n_t)).reshape(n_mu, n_row, n_col, n_t)
    return sim.astype(np.float32)


def synth_emg_for_muscle(muaps_static, spikes, time_samples):
    """Sum convolutions of each MU's MUAP with its spike train.

    muaps_static : [num_mus, n_row, n_col, n_t]   (single time step → broadcast to "1")
    """
    num_mus, n_row, n_col, n_t = muaps_static.shape
    emg = np.zeros((n_row, n_col, time_samples + n_t), dtype=np.float32)
    # generate_emg_mu expects shape [muap_steps, n_row, n_col, n_t]; treat static
    # MUAP as single step.
    muaps_step = muaps_static[:, None, :, :, :]   # [num_mus, 1, n_row, n_col, n_t]
    for mu in range(num_mus):
        emg += generate_emg_mu(muaps_step[mu], spikes[mu], time_samples).astype(np.float32)
    return emg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str,
                    default="demo_output/compute_stiffness/cup_task_stiffness_perturb_cmc.csv")
    ap.add_argument("--cfg", type=str, default="NeuroMotion/ckp/config.yaml")
    ap.add_argument("--model", type=str, default="NeuroMotion/ckp/model_linear.pth")
    ap.add_argument("--out_dir", type=str, default="demo_output/generate_emg_upper_limb")
    ap.add_argument("--device", type=str, default="mps", choices=["cpu", "mps", "cuda"],
                    help="Compute device. Default 'mps' (Apple Metal). Use 'cpu' if MPS misbehaves.")
    ap.add_argument("--num_mus_cap", type=int, default=None,
                    help="Cap NUM_MUS per muscle for fast testing (e.g. 30).")
    ap.add_argument("--duration", type=float, default=None,
                    help="Truncate input activations to this many seconds (default: full CSV)")
    ap.add_argument("--fs_in", type=float, default=100.0,
                    help="Sampling rate of compute_stiffness CSV (Hz)")
    ap.add_argument("--fs_out", type=int, default=2048,
                    help="EMG / spike-train sampling rate (Hz)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # MPS fallback: BioMime occasionally hits unsupported ops on MPS.
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("[warn] MPS unavailable, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"[info] device={device}")

    # --- 1. load activations ---
    csv_path = Path(args.csv)
    t_in, acts = load_activations(csv_path, UPPER_LIMB_MS)
    if args.duration is not None:
        m = t_in <= args.duration
        t_in = t_in[m]
        acts = {k: v[m] for k, v in acts.items()}
    duration = t_in[-1] - t_in[0]
    print(f"[info] loaded {csv_path.name}: {len(t_in)} samples, duration {duration:.2f} s")

    # --- 2. resample each activation to fs_out ---
    ext_dict = {ms: np.clip(resample_to_fs(a, args.fs_in, args.fs_out), 0, 1)
                for ms, a in acts.items()}
    time_samples = next(iter(ext_dict.values())).shape[0]
    print(f"[info] resampled to {args.fs_out} Hz -> {time_samples} samples")

    # --- 3. set up BioMime generator ---
    cfg = update_config(args.cfg)
    generator = Generator(cfg.Model.Generator)
    generator = load_generator(args.model, generator, args.device)
    generator.eval().to(device)
    latent_dim = cfg.Model.Generator.Latent

    # Butterworth low-pass for MUAP smoothing (matches mov2emg.py)
    b, a = butter(4, 800.0 / (0.5 * args.fs_out), btype="low", analog=False)

    # --- 4. per-muscle synthesis ---
    per_muscle_emg = {}
    n_row, n_col, n_t_muap = 10, 32, 96
    sum_emg = np.zeros((n_row, n_col, time_samples + n_t_muap), dtype=np.float32)

    t0 = time.time()
    for ms in UPPER_LIMB_MS:
        ext = ext_dict[ms]
        ts = time.time()
        pool, properties, num_mus = build_pool_for_muscle(
            ms, fs=args.fs_out, num_mus_cap=args.num_mus_cap)
        print(f"[{ms:9s}] N_MU={num_mus}  ext\u2208[{ext.min():.2f},{ext.max():.2f}]  pool init {time.time()-ts:.1f}s", flush=True)

        ts = time.time()
        _, spikes, _, _ = pool.generate_spike_trains(ext, fit=False)
        n_sp = sum(len(s) for s in spikes)
        print(f"[{ms:9s}]   spikes={n_sp}  ({time.time()-ts:.1f}s)", flush=True)

        ts = time.time()
        muaps = sample_static_muaps(generator, properties, num_mus,
                                    latent_dim, device, b, a)
        print(f"[{ms:9s}]   BioMime sample {muaps.shape}  ({time.time()-ts:.1f}s)", flush=True)

        ts = time.time()
        emg_ms = synth_emg_for_muscle(muaps, spikes, time_samples)
        print(f"[{ms:9s}]   EMG conv ({time.time()-ts:.1f}s)", flush=True)

        per_muscle_emg[ms] = emg_ms
        sum_emg += emg_ms

    print(f"[info] total synthesis time: {time.time() - t0:.1f}s")

    # --- 5. surface bank subsample ---
    bank8 = np.stack([sum_emg[r, c, :] for (r, c) in SURFACE_BANK_8CH], axis=0)  # [8, T]
    t_emg = np.arange(bank8.shape[1]) / args.fs_out

    # --- 6. save ---
    npz_path = out_dir / "emg_upper_limb.npz"
    np.savez_compressed(
        npz_path,
        t=t_emg.astype(np.float32),
        emg_8ch=bank8.astype(np.float32),
        emg_grid_sum=sum_emg,
        muscles=np.array(UPPER_LIMB_MS),
        ext_2048=np.stack([ext_dict[m] for m in UPPER_LIMB_MS], axis=0).astype(np.float32),
        bank_indices=np.array(SURFACE_BANK_8CH),
        fs=args.fs_out,
    )
    # per-muscle is too heavy to compress as one bundle — drop a few key ones
    for ms in ["BIClong", "TRIlong", "DELT2", "BRA"]:
        np.savez_compressed(out_dir / f"emg_per_muscle_{ms}.npz",
                            t=t_emg.astype(np.float32),
                            emg=per_muscle_emg[ms],
                            ext=ext_dict[ms].astype(np.float32))
    print(f"[info] saved {npz_path}")

    # --- 7. plot 8-channel surface EMG ---
    fig, axes = plt.subplots(8, 1, figsize=(10, 10), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t_emg, bank8[i], lw=0.4)
        ax.set_ylabel(f"ch{i+1}", rotation=0, ha="right", va="center")
        ax.tick_params(axis="y", labelsize=7)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Upper-limb synthetic surface EMG (8-channel bank)")
    fig.tight_layout()
    fig.savefig(out_dir / "emg_8ch.png", dpi=140)
    plt.close(fig)

    # plot activation envelopes vs |EMG| for a few muscles
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for ax, ms in zip(axes, ["BIClong", "TRIlong", "DELT2", "BRA"]):
        emg_ms = per_muscle_emg[ms]
        # representative channel: grid centre column
        sig = emg_ms[5, 16, :time_samples]
        env = np.abs(sig)
        # 50-ms RMS envelope
        win = max(1, int(0.05 * args.fs_out))
        env = np.convolve(env**2, np.ones(win)/win, mode="same") ** 0.5
        ax2 = ax.twinx()
        ax.plot(t_emg[:time_samples], ext_dict[ms], color="tab:orange", lw=1.2, label="activation")
        ax2.plot(t_emg[:time_samples], env, color="tab:blue", lw=0.6, alpha=0.8, label="|EMG| RMS")
        ax.set_ylabel(f"{ms}\nact", color="tab:orange")
        ax2.set_ylabel("EMG", color="tab:blue")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Activation (orange) vs synthetic EMG envelope (blue)")
    fig.tight_layout()
    fig.savefig(out_dir / "emg_vs_activation.png", dpi=140)
    plt.close(fig)

    print("[done]", out_dir)


if __name__ == "__main__":
    main()
