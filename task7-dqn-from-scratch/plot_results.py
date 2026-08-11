"""
Figures for the task 7 DQN runs.

    python plot_results.py                      # both runs, if both exist
    python plot_results.py --runs dqn2013       # just one
    python plot_results.py --compare-ppo ../task6-ppo-from-scratch/results/run1_episodes.csv

Reads the CSVs written by dqn_carracing.py and writes PNGs into
results/figures/. Missing runs are skipped with a warning rather than crashing,
so this is safe to run while training is still going.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "dqn2013": "#c0392b",   # red, the paper as published
    "dqn2015": "#2471a3",   # blue, with the target network
    "ppo": "#7d3c98",
}
LABELS = {
    "dqn2013": "DQN 2013 (no target network)",
    "dqn2015": "DQN + target network (Nature 2015)",
    "ppo": "PPO (task 6)",
}


def load(out_dir, run):
    """Return (episodes_df, updates_df) for a run, or (None, None) if absent."""
    ep_path = os.path.join(out_dir, f"{run}_episodes.csv")
    up_path = os.path.join(out_dir, f"{run}_updates.csv")
    ep = pd.read_csv(ep_path) if os.path.exists(ep_path) else None
    up = pd.read_csv(up_path) if os.path.exists(up_path) else None
    if ep is None:
        print(f"  (no episode log for {run}, skipping)")
    return ep, up


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_learning_curve(data, path):
    """The headline: per-episode reward and its 100-episode average."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for run, (ep, _) in data.items():
        if ep is None or ep.empty:
            continue
        c = COLORS.get(run, "#555")
        ax.plot(ep["step"], ep["reward"], color=c, alpha=0.15, linewidth=0.6)
        ax.plot(ep["step"], ep["running_avg"], color=c, linewidth=2.0,
                label=LABELS.get(run, run))

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    _style(ax, "environment steps (agent decisions)", "episode reward",
           "CarRacing-v3: episode reward, faint line per episode, bold = 100-episode average")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


def fig_vs_ppo(data, ppo, path):
    """
    DQN against last week's PPO on the same environment.

    Kept separate from the main learning curve because PPO ran to 2M steps and
    DQN to 400k, so plotting them on one axis squashes the DQN curves into the
    left margin and hides everything interesting about them.
    """
    fig, ax = plt.subplots(figsize=(10, 5.0))

    for run, (ep, _) in data.items():
        if ep is None or ep.empty:
            continue
        ax.plot(ep["step"], ep["running_avg"], color=COLORS.get(run, "#555"),
                linewidth=2.0, label=LABELS.get(run, run))

    ax.plot(ppo["step"], ppo["running_avg"], color=COLORS["ppo"],
            linewidth=1.8, linestyle="--", label=LABELS["ppo"])

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xscale("log")
    _style(ax, "environment steps, log scale", "100-episode average reward",
           "DQN this week against PPO last week, same environment and reward shaping")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


def fig_q_values(data, path):
    """
    Mean predicted Q over training.

    This is the figure that carries the argument. Without a target network the
    bootstrap target is produced by the same weights being updated, so any
    overestimate feeds straight back into the next target. If that happens, mean
    Q climbs away from anything the actual returns could justify. With a target
    network the same curve should stay far flatter.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    for run, (_, up) in data.items():
        if up is None or up.empty:
            continue
        c = COLORS.get(run, "#555")
        axes[0].plot(up["step"], up["mean_q"], color=c, linewidth=1.5,
                     label=LABELS.get(run, run))
        axes[1].plot(up["step"], up["loss"], color=c, linewidth=1.2, alpha=0.85,
                     label=LABELS.get(run, run))

    _style(axes[0], "environment steps", "mean predicted Q",
           "Mean Q of the taken action")
    _style(axes[1], "environment steps", "TD loss (Huber)", "Temporal-difference loss")
    axes[1].set_yscale("log")
    for a in axes:
        a.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


def fig_diagnostics(data, path):
    """Exploration schedule and throughput, the run's bookkeeping."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    for run, (ep, up) in data.items():
        c = COLORS.get(run, "#555")
        if up is not None and not up.empty:
            axes[0].plot(up["step"], up["epsilon"], color=c, linewidth=1.6,
                         label=LABELS.get(run, run))
            axes[1].plot(up["step"], up["steps_per_sec"], color=c, linewidth=1.2,
                         label=LABELS.get(run, run))

    _style(axes[0], "environment steps", "epsilon", "Exploration rate")
    _style(axes[1], "environment steps", "steps / sec", "Throughput")
    for a in axes:
        a.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


def summarize(data):
    """Print the numbers the write-up needs, so none of them get typed from memory."""
    print("\n" + "=" * 66)
    print("NUMBERS FOR THE WRITE-UP")
    print("=" * 66)
    for run, (ep, up) in data.items():
        if ep is None or ep.empty:
            continue
        best_i = ep["running_avg"].idxmax()
        print(f"\n{LABELS.get(run, run)}")
        print(f"  episodes                {len(ep)}")
        print(f"  total steps             {int(ep['step'].iloc[-1]):,}")
        print(f"  best avg100             {ep['running_avg'].max():.2f} "
              f"at step {int(ep.loc[best_i, 'step']):,}")
        print(f"  final avg100            {ep['running_avg'].iloc[-1]:.2f}")
        print(f"  best single episode     {ep['reward'].max():.2f}")
        print(f"  episodes scoring > 200  {int((ep['reward'] > 200).sum())}")
        if up is not None and not up.empty:
            print(f"  mean Q, first 10% of run  {up['mean_q'].head(max(1, len(up)//10)).mean():.2f}")
            print(f"  mean Q, last 10% of run   {up['mean_q'].tail(max(1, len(up)//10)).mean():.2f}")
            print(f"  max Q seen                {up['max_q'].max():.2f}")
            print(f"  mean throughput           {up['steps_per_sec'].mean():.1f} steps/sec")
    print("=" * 66 + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results")
    p.add_argument("--runs", nargs="+", default=["dqn2013", "dqn2015"])
    p.add_argument("--compare-ppo", default=None,
                   help="path to task 6 run1_episodes.csv")
    args = p.parse_args()

    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("loading runs...")
    data = {run: load(args.out_dir, run) for run in args.runs}
    if all(ep is None for ep, _ in data.values()):
        print("No run logs found. Has training been started?")
        return

    ppo = None
    if args.compare_ppo and os.path.exists(args.compare_ppo):
        ppo = pd.read_csv(args.compare_ppo)
        print(f"  loaded PPO comparison from {args.compare_ppo}")

    print("\nwriting figures...")
    fig_learning_curve(data, os.path.join(fig_dir, "01_learning_curve.png"))
    fig_q_values(data, os.path.join(fig_dir, "02_q_values.png"))
    fig_diagnostics(data, os.path.join(fig_dir, "03_diagnostics.png"))
    if ppo is not None and not ppo.empty:
        fig_vs_ppo(data, ppo, os.path.join(fig_dir, "04_dqn_vs_ppo.png"))

    summarize(data)


if __name__ == "__main__":
    main()
