#!/usr/bin/env bash
#
# Task 7, start to finish. One command:
#
#     cd ~/aiea-internship/task7-dqn-from-scratch && bash run_task7.sh
#
# It runs the correctness tests, then the two training runs, then the figures.
# Both runs are capped at 2.5 hours of wall clock each, so the whole thing ends
# in about five hours whatever the throughput turns out to be. If a run hits the
# cap early it still saves its checkpoint and logs, and the figures still work.
#
# To resume after an interruption:
#     python dqn_carracing.py --run-name dqn2013 --resume results/checkpoints/dqn2013_latest.pt --steps 400000
#
set -euo pipefail

STEPS="${STEPS:-400000}"
HOURS="${HOURS:-2.5}"
PY="${PY:-python -u}"   # -u so progress prints live through tee instead of buffering

cd "$(dirname "$0")"
mkdir -p results

echo "==> correctness tests"
# If these fail, nothing below is worth running.
$PY test_dqn.py

echo
echo "==> run A: DQN 2013 as published, no target network"
$PY dqn_carracing.py \
    --run-name dqn2013 \
    --steps "$STEPS" \
    --max-hours "$HOURS" \
    2>&1 | tee results/dqn2013_train.log

echo
echo "==> run B: same settings plus the Nature 2015 target network"
$PY dqn_carracing.py \
    --run-name dqn2015 \
    --target-net \
    --steps "$STEPS" \
    --max-hours "$HOURS" \
    2>&1 | tee results/dqn2015_train.log

echo
echo "==> evaluating both best checkpoints on unseen tracks"
$PY dqn_carracing.py --eval results/checkpoints/dqn2013_best.pt --episodes 5 \
    2>&1 | tee results/dqn2013_eval.log
$PY dqn_carracing.py --eval results/checkpoints/dqn2015_best.pt --episodes 5 \
    2>&1 | tee results/dqn2015_eval.log

echo
echo "==> figures"
$PY plot_results.py --compare-ppo ../task6-ppo-from-scratch/results/run1_episodes.csv

echo
echo "All done. Figures are in results/figures/."
echo "The numbers for the write-up are printed above under NUMBERS FOR THE WRITE-UP."
