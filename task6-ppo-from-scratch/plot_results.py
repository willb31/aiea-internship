"""
Turn the CSV logs from ppo_carracing.py into the figures for the write-up.

    python plot_results.py --run-name run1 --out-dir results

Produces, in results/figures/:
    01_learning_curve.png    episode reward and its 100-episode average
    02_losses.png            policy loss and value loss per update
    03_clipping.png          approximate KL and fraction of ratios being clipped
    04_diagnostics.png       entropy and explained variance
    summary.png              all four panels on one sheet
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#16191F"
ORANGE = "#E8590C"
SLATE = "#5A6472"
BLUE = "#1C7293"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 10,
    "axes.edgecolor": "#C7CCD3",
    "axes.labelcolor": INK,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": "#E8EBEF",
    "grid.linewidth": 0.8,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "legend.frameon": False,
})


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_learning_curve(ax, ep):
    ax.plot(ep["episode"], ep["reward"], color=SLATE, alpha=0.28, lw=0.8,
            label="episode reward")
    ax.plot(ep["episode"], ep["running_avg"], color=ORANGE, lw=2.0,
            label="100-episode average")
    # Only draw the "solved" line once the run is within sight of it. On a short
    # run it sits so far above the data that everything else flattens into a line.
    if ep["reward"].max() > 300:
        ax.axhline(900, color=INK, ls="--", lw=1.0, alpha=0.55)
        ax.text(ax.get_xlim()[1], 905, "  900 = solved", va="bottom", ha="right",
                fontsize=8.5, color=INK, alpha=0.7)
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.set_title("Learning curve")
    ax.legend(loc="upper left")
    tidy(ax)


def plot_losses(ax, up):
    ax.plot(up["step"], up["policy_loss"], color=ORANGE, lw=1.3, label="policy loss")
    ax.set_xlabel("environment step")
    ax.set_ylabel("policy loss", color=ORANGE)
    ax.tick_params(axis="y", labelcolor=ORANGE)

    ax2 = ax.twinx()
    ax2.plot(up["step"], up["value_loss"], color=BLUE, lw=1.3, label="value loss")
    ax2.set_ylabel("value loss", color=BLUE)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    ax2.grid(False)

    ax.set_title("Policy and value loss")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right")
    tidy(ax)
    ax2.spines["top"].set_visible(False)


def _robust(series, hi_pct=99.0):
    """
    Drop values too large to plot. An overflowing KL estimate can reach 1e35,
    which flattens every real value into the x-axis. Returns the masked series
    and the number of points that were dropped so the caller can say so.
    """
    s = pd.to_numeric(series, errors="coerce")
    finite = s[np.isfinite(s)]
    if finite.empty:
        return s, 0
    # A KL of 1.0 already means the policy moved enormously. Anything above
    # that is the estimator overflowing, not a measurement.
    cap = min(max(np.percentile(finite, hi_pct) * 5, 1e-6), 1.0)
    masked = s.where(np.isfinite(s) & (s <= cap))
    return masked, int((~masked.notna() & s.notna()).sum())


def plot_clipping(ax, up, clip_eps):
    ax.plot(up["step"], up["clip_fraction"], color=ORANGE, lw=1.4,
            label="fraction of ratios clipped")
    ax.set_xlabel("environment step")
    ax.set_ylabel("clip fraction", color=ORANGE)
    ax.tick_params(axis="y", labelcolor=ORANGE)
    ax.set_ylim(bottom=0)

    ax2 = ax.twinx()
    kl, dropped = _robust(up["approx_kl"])
    ax2.plot(up["step"], kl, color=BLUE, lw=1.4, label="approx KL")
    if dropped:
        ax2.text(0.02, 0.94, f"{dropped} KL value(s) off scale",
                 transform=ax2.transAxes, fontsize=8.5, color=BLUE, alpha=0.85)
    ax2.set_ylabel("approximate KL", color=BLUE)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    ax2.grid(False)

    ax.set_title(f"How often the clip is binding (eps = {clip_eps})")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right")
    tidy(ax)
    ax2.spines["top"].set_visible(False)


def plot_diagnostics(ax, up):
    l1, = ax.plot(up["step"], up["entropy"], color=ORANGE, lw=1.4,
                  label="policy entropy")
    ax.set_xlabel("environment step")
    ax.set_ylabel("entropy", color=ORANGE)
    ax.tick_params(axis="y", labelcolor=ORANGE)

    ax2 = ax.twinx()
    l2, = ax2.plot(up["step"], up["explained_variance"], color=BLUE, lw=1.4,
                   label="explained variance")
    ax2.axhline(0, color=SLATE, lw=0.8, alpha=0.5)  # deliberately unlabelled
    ax2.set_ylabel("explained variance", color=BLUE)
    ax2.set_ylim(-1.05, 1.05)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    ax2.grid(False)

    ax.set_title("Exploration and value-function fit")
    ax.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="lower right")
    tidy(ax)
    ax2.spines["top"].set_visible(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="run1")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--clip", type=float, default=0.1,
                   help="only used to label the clipping plot")
    args = p.parse_args()

    ep_path = os.path.join(args.out_dir, f"{args.run_name}_episodes.csv")
    up_path = os.path.join(args.out_dir, f"{args.run_name}_updates.csv")
    ep = pd.read_csv(ep_path)
    up = pd.read_csv(up_path)

    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    singles = [
        ("01_learning_curve", lambda ax: plot_learning_curve(ax, ep)),
        ("02_losses", lambda ax: plot_losses(ax, up)),
        ("03_clipping", lambda ax: plot_clipping(ax, up, args.clip)),
        ("04_diagnostics", lambda ax: plot_diagnostics(ax, up)),
    ]
    for name, draw in singles:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        draw(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{name}.png"), bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4))
    plot_learning_curve(axes[0, 0], ep)
    plot_losses(axes[0, 1], up)
    plot_clipping(axes[1, 0], up, args.clip)
    plot_diagnostics(axes[1, 1], up)
    fig.suptitle(
        f"PPO from scratch on CarRacing-v3  ·  run: {args.run_name}",
        fontsize=13, fontweight="bold", color=INK, y=0.995,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "summary.png"), bbox_inches="tight")
    plt.close(fig)

    # a few numbers worth quoting in the write-up
    best_avg = ep["running_avg"].max()
    last50 = ep["reward"].tail(50).mean()
    print(f"episodes:              {len(ep)}")
    print(f"environment steps:     {ep['step'].iloc[-1]:,}")
    print(f"updates:               {len(up)}")
    print(f"best 100-ep average:   {best_avg:.1f}")
    print(f"mean of last 50 eps:   {last50:.1f}")
    print(f"peak single episode:   {ep['reward'].max():.1f}")
    print(f"\nfigures written to {fig_dir}/")


if __name__ == "__main__":
    main()
