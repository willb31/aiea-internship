#!/usr/bin/env bash
#
# Task 8, end to end. Tests, both training runs, evaluation, figures.
#
#   bash run_task8.sh
#
# The two runs are sequential and take roughly three hours each, so budget
# around six hours. Everything is checkpointed every 25k steps, so if a run dies
# you can pick it up with --resume rather than starting over.
#
# Note there is no --max-hours here, on purpose. Task 7 capped on wall clock as
# well as steps, the target-network run hit the clock first, and the comparison
# ended up being between one run at 400k steps and another at 329k. Both runs
# this week stop at exactly 400,000 steps and nowhere else.

set -euo pipefail

STEPS=${STEPS:-400000}
SEED=${SEED:-0}
EVAL_EPISODES=${EVAL_EPISODES:-10}

cd "$(dirname "$0")"

echo "=============================================================="
echo " Task 8: Double DQN refinement"
echo " steps per run: $STEPS    seed: $SEED    eval episodes: $EVAL_EPISODES"
echo "=============================================================="

# ---------------------------------------------------------------- tests
echo
echo ">>> correctness checks"
python test_ddqn.py

# ---------------------------------------------------------------- run A
# The baseline: target network, standard max target. Same as task 7's dqn2015
# but run to a full 400k steps so the comparison against run B is clean.
echo
echo ">>> run A of 2: baseline, target network, standard target  (~3 h)"
python ddqn_carracing.py \
    --run-name dqn2015_full \
    --steps "$STEPS" \
    --seed "$SEED" \
    --target-net \
    2>&1 | tee results/dqn2015_full_train.log

# ---------------------------------------------------------------- run B
# The refinement. Identical to run A in every respect except --double.
echo
echo ">>> run B of 2: Double DQN  (~3 h)"
python ddqn_carracing.py \
    --run-name ddqn \
    --steps "$STEPS" \
    --seed "$SEED" \
    --target-net \
    --double \
    2>&1 | tee results/ddqn_train.log

# ---------------------------------------------------------------- eval
# Greedy rollouts on tracks neither run saw during training. Task 7 used 5
# episodes and got a standard deviation of 229, which is too noisy to separate
# two runs that might genuinely differ by 50 points. 10 here.
echo
echo ">>> evaluating both best checkpoints on $EVAL_EPISODES unseen tracks"
for run in dqn2015_full ddqn; do
    echo "  --- $run"
    python ddqn_carracing.py \
        --eval "results/checkpoints/${run}_best.pt" \
        --episodes "$EVAL_EPISODES" \
        --seed "$SEED" \
        2>&1 | tee "results/${run}_eval.log"
done

# ---------------------------------------------------------------- figures
echo
echo ">>> figures and summary numbers"
python plot_results.py

echo
echo "=============================================================="
echo " Done. Figures in results/figures/, numbers printed above."
echo "=============================================================="
