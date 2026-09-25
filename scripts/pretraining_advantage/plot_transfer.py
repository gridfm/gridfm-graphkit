"""Plot the section 5.5.3 scratch vs finetune figure from the committed CSV."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FONT_RC = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def plot(csv_path: Path, output_path: Path) -> None:
    plt.rcParams.update(FONT_RC)
    df = pd.read_csv(csv_path)
    zeroshot = df[(df["type"] == "finetune") & (df["n_scenarios"] == 0)]
    main = df[~((df["type"] == "finetune") & (df["n_scenarios"] == 0))]
    dc = df["dc_avg_active_res_mw"].to_numpy(dtype=float)
    if not np.allclose(dc, dc[0], rtol=1e-9, atol=1e-12):
        raise ValueError("DC residuals are not constant")

    scenarios = sorted(main["n_scenarios"].unique())

    def values(run_type: str) -> np.ndarray:
        sub = main[main["type"] == run_type].set_index("n_scenarios")
        return np.array([sub.loc[s, "ac_avg_active_res_mw"] for s in scenarios])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(scenarios, values("scratch"), marker="o", color="tab:blue", label="Trained from scratch")
    ax.plot(scenarios, values("finetune"), marker="o", color="tab:pink", label="Finetuned")
    if not zeroshot.empty:
        ax.axhline(float(zeroshot["ac_avg_active_res_mw"].mean()), linestyle="--", linewidth=2, color="black", label="Zero-shot")
    ax.axhline(float(dc[0]), linestyle="-.", linewidth=2, color="tab:orange", label="DC-PF")
    ax.set_xlabel("Training scenario count [-]", fontsize=20)
    ax.set_ylabel("Mean active power balance residual [MW]", fontsize=20)
    ax.set_xticks(scenarios)
    ax.set_xticklabels([str(s) for s in scenarios], rotation=45, ha="right")
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(axis="both", linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    plot(root / "results" / "transfer_case118.csv", root / "figures" / "scratch_vs_finetune_active_residuals.pdf")
    print(f"Saved {root / 'figures' / 'scratch_vs_finetune_active_residuals.pdf'}")
