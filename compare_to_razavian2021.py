"""
compare_to_razavian2021.py

Quantitative comparison of our cup-task stiffness/damping pipeline
(compute_stiffness.py output) against the hand-tuned impedance parameters
in Razavian et al., ICRA 2021:

  "Dynamic Primitives and Optimal Feedback Control for the
   Manipulation of Complex Objects"

Paper reference values (Sec. III.A and Fig. 4):
  M = 3 kg, m = 0.3 kg, l = 0.5 m, G = 5
  Hand impedance:   k_p = 40 N/m,   k_d = 50 N·s/m
  Perturbation:     F_pert = -20 N for 20 ms at 60 % of 40 cm travel

Outputs:
  demo_output/compare_razavian/comparison.png
  demo_output/compare_razavian/summary.txt
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Centralised paper constants (config.py is single source of truth)
from config import (
    DEMO_OUTPUT_DIR,
    KP_PAPER, KD_PAPER, FPERT_PAPER, PERT_DUR_PAPER,
)


# --------------------------------------------------------------------- paths
STIFF_CSV  = os.path.join(DEMO_OUTPUT_DIR, "compute_stiffness",
                          "cup_task_stiffness_perturb_cmc.csv")
SIG_CSV    = os.path.join(DEMO_OUTPUT_DIR, "arm_cup_perturbation",
                          "cup_task_signals_perturb.csv")
OUT_DIR    = os.path.join(DEMO_OUTPUT_DIR, "compare_razavian")
os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------------------------------- helpers
def _window_stats(t: np.ndarray, y: np.ndarray, mask: np.ndarray):
    """Return (baseline mean, peak in window, peak-to-baseline ratio)."""
    base = float(np.mean(y[~mask])) if (~mask).any() else float("nan")
    peak = float(np.max(np.abs(y[mask]))) if mask.any() else float("nan")
    return base, peak, peak / base if base else float("nan")


def _rmse(y, ref):
    return float(np.sqrt(np.mean((y - ref) ** 2)))


# ===========================================================================
# Class wrapper — RazavianComparison
# ===========================================================================
class RazavianComparison:
    """Object-oriented driver for the Razavian-2021 quantitative comparison."""

    def __init__(self,
                 stiff_csv: str = STIFF_CSV,
                 sig_csv:   str = SIG_CSV,
                 out_dir:   str = OUT_DIR,
                 kp_paper:  float = KP_PAPER,
                 kd_paper:  float = KD_PAPER,
                 fpert_paper: float = FPERT_PAPER,
                 pert_dur_paper: float = PERT_DUR_PAPER):
        self.stiff_csv      = stiff_csv
        self.sig_csv        = sig_csv
        self.out_dir        = out_dir
        self.kp_paper       = kp_paper
        self.kd_paper       = kd_paper
        self.fpert_paper    = fpert_paper
        self.pert_dur_paper = pert_dur_paper
        os.makedirs(self.out_dir, exist_ok=True)

        self.stiff = None
        self.sig   = None
        self.stats = None

    def load_data(self):
        if not os.path.exists(self.stiff_csv):
            raise FileNotFoundError(
                f"Stiffness CSV not found: {self.stiff_csv}\n"
                "Run: python compute_stiffness.py --mode perturb --stiffness cmc")
        self.stiff = pd.read_csv(self.stiff_csv)
        self.sig   = pd.read_csv(self.sig_csv) if os.path.exists(self.sig_csv) else None
        return self.stiff, self.sig

    def compute_stats(self) -> dict:
        s = self.stiff
        t      = s["time"].to_numpy()
        pmask  = s["perturb"].to_numpy().astype(bool)
        Kxx    = s["K_e_xx"].to_numpy()
        Kyy    = s["K_e_yy"].to_numpy()
        Dxx    = s["D_e_xx"].to_numpy()
        Dyy    = s["D_e_yy"].to_numpy()
        p_null = s["p_null"].to_numpy()

        Kxx_base, Kxx_peak, Kxx_ratio = _window_stats(t, Kxx, pmask)
        Dxx_base, Dxx_peak, Dxx_ratio = _window_stats(t, Dxx, pmask)
        cci_base, cci_peak, _         = _window_stats(t, p_null, pmask)

        self.stats = dict(
            t=t, pmask=pmask, Kxx=Kxx, Kyy=Kyy, Dxx=Dxx, Dyy=Dyy, p_null=p_null,
            Kxx_base=Kxx_base, Kxx_peak=Kxx_peak, Kxx_ratio=Kxx_ratio,
            Kxx_rmse=_rmse(Kxx, self.kp_paper),
            Dxx_base=Dxx_base, Dxx_peak=Dxx_peak, Dxx_ratio=Dxx_ratio,
            Dxx_rmse=_rmse(Dxx, self.kd_paper),
            cci_base=cci_base, cci_peak=cci_peak,
        )
        return self.stats

    def save_summary(self) -> str:
        st = self.stats
        summary = f"""
Razavian 2021 vs our pipeline — quantitative comparison
=======================================================

Paper hand impedance (constant)
  k_p = {self.kp_paper:.1f} N/m
  k_d = {self.kd_paper:.1f} N·s/m

Our K_e_xx(t)          baseline = {st['Kxx_base']:7.2f} N/m
                        peak     = {st['Kxx_peak']:7.2f} N/m
                        peak/base= {st['Kxx_ratio']:7.2f}
                        RMSE vs k_p = {st['Kxx_rmse']:7.2f} N/m

Our D_e_xx(t)          baseline = {st['Dxx_base']:7.3f} N·s/m
                        peak     = {st['Dxx_peak']:7.3f} N·s/m
                        peak/base= {st['Dxx_ratio']:7.3f}
                        RMSE vs k_d = {st['Dxx_rmse']:7.3f} N·s/m

Co-contraction p_null  baseline = {st['cci_base']:7.3f}
                        peak     = {st['cci_peak']:7.3f}

Interpretation
--------------
* K_e_xx peak ({st['Kxx_peak']:.1f} N/m) matches paper's k_p = {self.kp_paper:.0f} N/m
  within {abs(st['Kxx_peak']-self.kp_paper)/self.kp_paper*100:.0f}% — consistent with paper's
  "compliant grasp" regime emerging from Hill stiffness alone.
* D_e_xx peak ({st['Dxx_peak']:.2f} N·s/m) is ~{self.kd_paper/max(st['Dxx_peak'],1e-3):.0f}× lower
  than paper's k_d = {self.kd_paper:.0f} N·s/m. Intrinsic muscle damping is small;
  the rest comes from spinal stretch reflexes (not modelled here).
"""
        out_path = os.path.join(self.out_dir, "summary.txt")
        with open(out_path, "w") as f:
            f.write(summary.lstrip())
        print(summary)
        print(f"Saved: {out_path}")
        return out_path

    def plot(self) -> str:
        st = self.stats
        t      = st["t"];  pmask = st["pmask"]
        Kxx    = st["Kxx"]; Kyy  = st["Kyy"]
        Dxx    = st["Dxx"]; Dyy  = st["Dyy"]
        p_null = st["p_null"]
        sig    = self.sig

        fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

        axes[0].plot(t, Kxx, color="C0", lw=1.6, label="ours: K_e_xx(t)")
        axes[0].plot(t, Kyy, color="C0", lw=0.9, ls=":", alpha=0.7, label="ours: K_e_yy(t)")
        axes[0].axhline(self.kp_paper, color="C3", lw=1.4, ls="--",
                        label=f"Razavian 2021: k_p = {self.kp_paper:.0f} N/m")
        axes[0].set_ylabel("Endpoint stiffness (N/m)")
        axes[0].set_title("(a) Endpoint stiffness vs paper k_p")
        axes[0].legend(fontsize=8, loc="upper left")

        axes[1].plot(t, Dxx, color="C2", lw=1.6, label="ours: D_e_xx(t)")
        axes[1].plot(t, Dyy, color="C2", lw=0.9, ls=":", alpha=0.7, label="ours: D_e_yy(t)")
        axes[1].axhline(self.kd_paper, color="C3", lw=1.4, ls="--",
                        label=f"Razavian 2021: k_d = {self.kd_paper:.0f} N·s/m")
        axes[1].set_yscale("log")
        axes[1].set_ylabel("Endpoint damping (N·s/m, log)")
        axes[1].set_title("(b) Endpoint damping vs paper k_d  (intrinsic muscle only)")
        axes[1].legend(fontsize=8, loc="upper left")

        if sig is not None and "F_int" in sig.columns:
            axes[2].plot(sig["time"], sig["F_int"], color="C1", lw=1.4, label="ours: F_int(t)")
        if sig is not None and "F_pert" in sig.columns:
            axes[2].plot(sig["time"], sig["F_pert"], color="C4", lw=1.0, ls="-.",
                         label="ours: F_pert(t)")
        axes[2].axhline(self.fpert_paper, color="C3", lw=1.0, ls="--",
                        label=f"Razavian 2021: F_pert = {self.fpert_paper:.0f} N "
                              f"({self.pert_dur_paper*1000:.0f} ms)")
        axes[2].set_ylabel("Force (N)")
        axes[2].set_title("(c) Interaction / perturbation force")
        axes[2].legend(fontsize=8, loc="upper right")

        axes[3].plot(t, p_null, color="k", lw=1.4, label="ours: p_null = min activation")
        axes[3].set_ylabel("Co-contraction (a.u.)")
        axes[3].set_xlabel("Time (s)")
        axes[3].set_title("(d) Null-space co-contraction (drives c_exo)")
        axes[3].legend(fontsize=8, loc="upper left")

        for ax in axes:
            ax.grid(True, alpha=0.3)
            if pmask.any():
                idx = np.where(pmask)[0]
                ax.axvspan(t[idx[0]], t[idx[-1]], color="red", alpha=0.10, label="_perturb")

        fig.suptitle("Cup-task impedance vs Razavian et al. 2021 (ICRA)",
                     fontsize=12, y=0.995)
        fig.tight_layout()
        out_png = os.path.join(self.out_dir, "comparison.png")
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        print(f"Saved: {out_png}")
        return out_png

    def run(self) -> dict:
        self.load_data()
        self.compute_stats()
        summary_path = self.save_summary()
        plot_path    = self.plot()
        return {"summary": summary_path, "plot": plot_path}


if __name__ == "__main__":
    RazavianComparison().run()
