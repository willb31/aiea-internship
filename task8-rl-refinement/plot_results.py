"""
Figures for the task 8 Double DQN runs.

    python plot_results.py                       # both task 8 runs plus task 7 for reference
    python plot_results.py --runs ddqn           # just one
    python plot_results.py --no-task7            # task 8 runs only

Reads the CSVs written by ddqn_carracing.py and writes PNGs into
results/figures/. Missing runs are skipped with a warning rather than crashing,
so this is safe to run while training is still going.

It also prints every number the write-up quotes, so the write-up never has a
figure in it that nobody checked against the data.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "ddqn": "#1e8449",           # green, this week's refinement
    "dqn2015_full": "#2471a3",   # blue, this week's baseline
    "dqn2015": "#85c1e9",        # pale blue, task 7's truncated baseline
    "dqn2013": "#c0392b",        # red, task 7's no-target-network run
}
LABELS = {
    "ddqn": "Double DQN (task 8)",
    "dqn2015_full": "DQN + target network, 400k (task 8 baseline)",
    "dqn2015": "DQN + target network, 329k (task 7)",
    "dqn2013": "DQN 2013, no target network (task 7)",
}
TASK7_DIR = "../task7-dqn-from-scratch/results"


def load(out_dir, run):
    """Return (episodes, updates, holdout) for a run. Missing pieces come back None."""
    def _read(suffix):
        p = os.path.join(out_dir, f"{run}_{suffix}.csv")
        return pd.read_csv(p) if os.path.exists(p) else None

    ep, up, ho = _read("episodes"), _read("updates"), _read("holdout")
    if ep is None:
        print(f"  (no episode log for {run}, skipping)")
    return ep, up, ho


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# --------------------------------------------------------------------------
def fig_learning_curve(data, path):
    """The headline. Faint per-episode scores, solid 100-episode average."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for run, (ep, _, _) in data.items():
        if ep is None or ep.empty:
            continue
        c = COLORS.get(run, "#555555")
        ax.plot(ep["step"], ep["reward"], color=c, alpha=0.10, linewidth=0.6)
        ax.plot(ep["step"], ep["running_avg"], color=c, linewidth=2.0,
                label=LABELS.get(run, run))

    _style(ax, "environment step (agent decisions)", "episode return",
           "Learning curves, 100-episode running average")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_holdout_values(data, path):
    """
    Mean max-Q over the 500 states frozen after warm-up, the Nature paper's
    Figure 2 metric, plotted against the return being achieved at the same
    moment on a second axis.

    Two axes, not one, and this matters. The value estimate is a *discounted*
    sum with gamma = 0.99, which over a ~250-decision episode is worth about 92
    effective steps. The episode return is undiscounted. Plotting them on a
    shared axis and reading the difference as bias would be measuring the
    discount factor, not the algorithm.
    """
    runs = [r for r in ("dqn2015_full", "ddqn") if data.get(r, (None,))[0] is not None]
    if not runs:
        print("  (no task 8 runs with a held-out trace, skipping)")
        return

    fig, axes = plt.subplots(1, len(runs), figsize=(5.8 * len(runs), 4.8), squeeze=False)

    for ax, run in zip(axes[0], runs):
        _, _, ho = data[run]
        if ho is None or ho.empty:
            ax.text(0.5, 0.5, f"no held-out trace for {run}", ha="center")
            continue
        c = COLORS.get(run, "#555555")
        ax.plot(ho["step"], ho["holdout_mean_value"], color=c, linewidth=2.0,
                label="predicted value (discounted)")
        ax.set_ylabel("predicted value, held-out states", color=c)
        ax.tick_params(axis="y", labelcolor=c)

        ax2 = ax.twinx()
        rr = pd.to_numeric(ho["recent_return"], errors="coerce")
        ax2.plot(ho["step"], rr, color="#7f8c8d", linewidth=1.5, linestyle="--",
                 label="episode return (undiscounted)")
        ax2.set_ylabel("episode return, 100-ep avg", color="#7f8c8d")
        ax2.tick_params(axis="y", labelcolor="#7f8c8d")
        ax2.spines["top"].set_visible(False)

        _style(ax, "environment step", "predicted value, held-out states",
               LABELS.get(run, run).split(" (")[0])
        ax.set_ylabel("predicted value, held-out states", color=c)

    fig.suptitle("Held-out value estimate and achieved return. Separate axes: the "
                 "value is discounted, the return is not.", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_value_ratio(data, path):
    """
    The actual head-to-head on bias.

    Absolute overestimation cannot be read off these runs: it would need the
    true discounted return from each held-out state under the current policy,
    and CarRacing cannot be reset to a mid-episode state to measure that. What
    *can* be compared is the two runs against each other, because both are
    measured the identical way at the identical steps and both end up achieving
    almost the same return. So the value estimate is normalised by the return
    being achieved at that moment, and the two normalised traces are compared.

    The units of the ratio are arbitrary. Only the gap between the two lines
    means anything, and it means: for the same achieved performance, this
    algorithm's value estimates are higher or lower.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    stats = {}

    for run, (_, _, ho) in data.items():
        if ho is None or ho.empty:
            continue
        h = ho[pd.to_numeric(ho["recent_return"], errors="coerce") > 50]
        if h.empty:
            continue
        ratio = h["holdout_mean_value"] / pd.to_numeric(h["recent_return"], errors="coerce")
        ax.plot(h["step"], ratio, color=COLORS.get(run, "#555555"), linewidth=2.0,
                label=LABELS.get(run, run))
        stats[run] = (h["step"].values, ratio.values)

    if not stats:
        print("  (no held-out traces, skipping the value ratio figure)")
        plt.close(fig)
        return

    _style(ax, "environment step", "held-out value  /  return achieved",
           "Value estimate per unit of achieved performance.\n"
           "Lower means the same driving is being valued less optimistically.")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_eval(eval_scores, path):
    """Evaluation on unseen tracks, mean with a standard-deviation bar."""
    if not eval_scores:
        print("  (no eval logs found, skipping the eval figure)")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5))
    runs = list(eval_scores)
    means = [np.mean(eval_scores[r]) for r in runs]
    stds = [np.std(eval_scores[r]) for r in runs]
    colors = [COLORS.get(r, "#555555") for r in runs]

    xs = np.arange(len(runs))
    ax.bar(xs, means, yerr=stds, capsize=6, color=colors, alpha=0.85,
           error_kw={"linewidth": 1.4})
    # the individual episodes behind each bar, so the spread is visible
    for x, r in zip(xs, runs):
        pts = eval_scores[r]
        ax.scatter(np.full(len(pts), x) + np.random.uniform(-0.09, 0.09, len(pts)),
                   pts, color="#2c3e50", s=16, zorder=3, alpha=0.7)

    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS.get(r, r).split(" (")[0] for r in runs],
                       rotation=12, ha="right", fontsize=8)
    _style(ax, "", "return on unseen tracks",
           "Greedy evaluation, unseen tracks (dots are individual episodes)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_diagnostics(data, path):
    """Loss, target magnitude and throughput. The things that explain a weird curve."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for run, (_, up, _) in data.items():
        if up is None or up.empty:
            continue
        c = COLORS.get(run, "#555555")
        lbl = LABELS.get(run, run)
        axes[0].plot(up["step"], up["loss"].rolling(20, min_periods=1).mean(),
                     color=c, linewidth=1.6, label=lbl)
        if "mean_y" in up.columns:
            axes[1].plot(up["step"], up["mean_y"].rolling(20, min_periods=1).mean(),
                         color=c, linewidth=1.6, label=lbl)
        axes[2].plot(up["step"], up["steps_per_sec"], color=c, linewidth=1.6, label=lbl)

    axes[0].set_yscale("log")
    _style(axes[0], "step", "Huber loss (smoothed)", "Loss")
    _style(axes[1], "step", "mean bootstrap target y", "Target magnitude")
    _style(axes[2], "step", "steps / second", "Throughput")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
def read_eval_log(path):
    """Pull the per-episode scores out of an eval log."""
    if not os.path.exists(path):
        return None
    scores = []
    with open(path) as fh:
        for line in fh:
            m = re.search(r"eval episode\s+\d+:\s*(-?[\d.]+)", line)
            if m:
                scores.append(float(m.group(1)))
    return scores or None


def summarise(data, eval_scores):
    """Print every number the write-up quotes."""
    print("\n" + "=" * 74)
    print("NUMBERS FOR THE WRITE-UP")
    print("=" * 74)

    for run, (ep, up, ho) in data.items():
        if ep is None or ep.empty:
            continue
        print(f"\n{LABELS.get(run, run)}  [{run}]")
        print(f"  episodes                {len(ep)}")
        print(f"  steps                   {int(ep['step'].iloc[-1]):,}")
        print(f"  best 100-ep average     {ep['running_avg'].max():.2f}")
        print(f"  final 100-ep average    {ep['running_avg'].iloc[-1]:.2f}")
        print(f"  best single episode     {ep['reward'].max():.2f}")
        last200 = ep["reward"].tail(200)
        print(f"  last 200 episodes       mean {last200.mean():.2f}  std {last200.std():.2f}")

        if ho is not None and not ho.empty:
            h = ho[pd.to_numeric(ho["recent_return"], errors="coerce") > 50]
            print(f"  final held-out value    {ho['holdout_mean_value'].iloc[-1]:.2f}")
            print(f"  peak held-out value     {ho['holdout_mean_value'].max():.2f}")
            if not h.empty:
                ratio = h["holdout_mean_value"] / pd.to_numeric(h["recent_return"],
                                                                errors="coerce")
                print(f"  value / return          mean {ratio.mean():.4f}  "
                      f"final {ratio.iloc[-1]:.4f}")

        if up is not None and not up.empty:
            print(f"  mean throughput         {up['steps_per_sec'].iloc[-1]:.1f} steps/s")

        if run in eval_scores:
            s = eval_scores[run]
            print(f"  EVAL on unseen tracks   mean {np.mean(s):.2f}  std {np.std(s):.2f}  "
                  f"over {len(s)} episodes")

    # the head-to-head the deliverable is actually about
    a, b = data.get("dqn2015_full"), data.get("ddqn")
    if a and b and a[0] is not None and b[0] is not None:
        base, dbl = a[0]["running_avg"].max(), b[0]["running_avg"].max()
        print("\n" + "-" * 74)
        print("HEAD TO HEAD  (Double DQN vs the 400k target-network baseline)")
        print(f"  best 100-ep average   {base:.2f}  ->  {dbl:.2f}   "
              f"({dbl - base:+.2f}, {100 * (dbl - base) / abs(base):+.1f}%)")
        if "dqn2015_full" in eval_scores and "ddqn" in eval_scores:
            eb, ed = np.mean(eval_scores["dqn2015_full"]), np.mean(eval_scores["ddqn"])
            print(f"  eval on unseen tracks {eb:.2f}  ->  {ed:.2f}   "
                  f"({ed - eb:+.2f}, {100 * (ed - eb) / abs(eb):+.1f}%)")
        if a[2] is not None and b[2] is not None:
            # Paired on matching steps. Both runs are sampled at the same steps
            # and end up at almost the same return, so the difference in the
            # normalised value estimate is the bias difference.
            m = a[2].merge(b[2], on="step", suffixes=("_base", "_dbl"))
            m = m[(pd.to_numeric(m["recent_return_base"], errors="coerce") > 50)
                  & (pd.to_numeric(m["recent_return_dbl"], errors="coerce") > 50)]
            if not m.empty:
                ra = m["holdout_mean_value_base"] / m["recent_return_base"]
                rb = m["holdout_mean_value_dbl"] / m["recent_return_dbl"]
                diff = rb - ra
                print(f"  value / return        {ra.mean():.4f}  ->  {rb.mean():.4f}   "
                      f"({diff.mean():+.4f})")
                print(f"    double is lower at  {100 * (diff < 0).mean():.0f}% "
                      f"of {len(m)} matched checkpoints")
                print("    NOTE: consecutive checkpoints inside one run are heavily")
                print("    autocorrelated and this is a single seed, so treat this as")
                print("    descriptive. It is not an independent-sample significance test.")
    print("=" * 74 + "\n")


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="figures for the task 8 Double DQN runs")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--runs", nargs="*", default=["dqn2015_full", "ddqn"])
    p.add_argument("--no-task7", action="store_true",
                   help="skip the task 7 runs that are plotted for reference")
    p.add_argument("--fig-dir", default=None)
    args = p.parse_args()

    fig_dir = args.fig_dir or os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    data, eval_scores = {}, {}

    for run in args.runs:
        data[run] = load(args.out_dir, run)
        s = read_eval_log(os.path.join(args.out_dir, f"{run}_eval.log"))
        if s:
            eval_scores[run] = s

    if not args.no_task7 and os.path.isdir(TASK7_DIR):
        for run in ("dqn2013", "dqn2015"):
            data[run] = load(TASK7_DIR, run)
            s = read_eval_log(os.path.join(TASK7_DIR, f"{run}_eval.log"))
            if s:
                eval_scores[run] = s

    print(f"writing figures to {fig_dir}")
    fig_learning_curve(data, os.path.join(fig_dir, "01_learning_curve.png"))
    fig_holdout_values(data, os.path.join(fig_dir, "02_holdout_values.png"))
    fig_value_ratio(data, os.path.join(fig_dir, "03_value_ratio.png"))
    fig_eval(eval_scores, os.path.join(fig_dir, "04_eval_unseen_tracks.png"))
    fig_diagnostics(data, os.path.join(fig_dir, "05_diagnostics.png"))

    summarise(data, eval_scores)


if __name__ == "__main__":
    main()
