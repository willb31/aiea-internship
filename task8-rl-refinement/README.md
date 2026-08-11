# Task 8: Double DQN refinement on discretized CarRacing-v3

Task 7 built DQN from scratch and ran it twice, once as the 2013 paper published
it and once with the Nature 2015 target network. The target network won. The
diagnostic that explained *why* it won is the starting point for this week.

## What task 7 measured, and why it points here

The run without a target network predicted a mean Q of **474.8** while actually
scoring **659.5**. Its value estimates were badly out of step with what it was
achieving. That is not a bug, it is a structural property of Q-learning: the
target is

    y = r + gamma * max_a' Q(s', a')

and that `max` is taken over the same noisy estimates it is selecting from. Any
action whose value happens to be overestimated is exactly the action the max
picks, so the error does not average out over many updates, it accumulates.

Double Q-learning (van Hasselt, Guez and Silver 2016,
[arXiv:1509.06461](https://arxiv.org/abs/1509.06461)) splits the two jobs the max
is doing. One network decides which action is best, a different one says what it
is worth:

| | target |
| --- | --- |
| standard | `y = r + γ · Q_target(s', argmax_a' Q_target(s', a'))` |
| double | `y = r + γ · Q_target(s', argmax_a' Q_online(s', a'))` |

An action now has to be overestimated by *both* networks to survive into the
target, which is much rarer than being overestimated by one.

## The experiment

Two runs, 400,000 steps each, seed 0. Identical hyperparameters, identical
network, identical optimizer, identical environment wrapper. `env_and_replay.py`
is byte-identical to the task 7 file. The only thing that differs between the two
runs is the branch inside `bellman_target()`.

| run | target network | double | steps | best avg100 | eval, 10 unseen tracks |
| --- | --- | --- | --- | --- | --- |
| `dqn2015_full` | yes | no | 400,000 | 782.92 | 846.74 ± 73.39 |
| `ddqn` | yes | **yes** | 400,000 | 786.23 | **867.42 ± 52.32** |

Returns are close to tied. The difference shows in the value estimates, where
Double DQN is the lower of the two at 85% of matched checkpoints. Full discussion
in `DELIVERABLE.md`.

Rerunning the task 7 baseline to its full 400k steps mattered more than the
algorithm did: on identical evaluation seeds it went from 688.49 to 846.74, so
task 7's wall-clock cap had been understating it by 23%.

## Two methodology fixes carried over from task 7

**Runs are capped by step count only.** Task 7 capped on wall clock as well, the
target-network run hit the clock first, and the comparison ended up being between
one run at 400k steps and another at 329k. `run_task8.sh` passes no
`--max-hours` and a test asserts the default stays `None`.

**Value estimates come from a fixed held-out set of states.** Task 7 read mean Q
off the training minibatch, which drifts with wherever the current policy happens
to be driving, so it confounds "the estimate got bigger" with "the agent went
somewhere different". This week 500 states are frozen right after the warm-up and
never change again, which is how the Nature paper's Figure 2 does it. The trace
goes to `{run}_holdout.csv` alongside the return being achieved at the same
moment.

One warning about reading that file, because I got it wrong first. The value is a
**discounted** sum at gamma 0.99, worth roughly 92 effective steps over a
250-decision episode. `recent_return` is **undiscounted**. Subtracting one from
the other and calling it overestimation measures the discount factor, not the
algorithm, and produces a large negative number that looks like severe
underestimation. An absolute overestimation figure would need the true discounted
return from each held-out state under the current policy, and CarRacing cannot be
reset to a mid-episode state to get it. What is valid is comparing the two runs
to each other, since both are measured identically at identical steps, which is
what `fig_value_ratio` in `plot_results.py` does.

## Running it

```bash
pip install swig
pip install -r requirements.txt
bash run_task8.sh
```

That runs the tests, both training runs, a 10-episode evaluation of each best
checkpoint on unseen tracks, and the figures. Roughly six hours total, two
sequential three-hour runs. Checkpoints are written every 25k steps, so a run
that dies can be picked up with `--resume` instead of restarted.

Individually:

```bash
python test_ddqn.py                                    # ~30 seconds, run this first
python ddqn_carracing.py --run-name dqn2015_full --steps 400000 --target-net
python ddqn_carracing.py --run-name ddqn --steps 400000 --target-net --double
python ddqn_carracing.py --eval results/checkpoints/ddqn_best.pt --episodes 10
python plot_results.py
```

Evaluation uses `seed + 10000`, so the tracks are ones neither run trained on.
Task 7 evaluated on 5 episodes and got a standard deviation of 229, too noisy to
separate two runs that might genuinely differ by 50 points. This week uses 10.

## Files

| file | what it is |
| --- | --- |
| `env_and_replay.py` | unchanged from task 7. Discrete action set, CarRacing wrapper, replay memory, epsilon schedule. |
| `ddqn_carracing.py` | Q-network, `bellman_target()`, training loop, held-out value trace, evaluation, CLI. |
| `test_ddqn.py` | correctness checks. Run before any long run. |
| `plot_results.py` | figures, plus it prints every number the write-up quotes. |
| `run_task8.sh` | the whole week end to end. |

## A note on testing the Double DQN change

A wrong Double DQN does not crash. It trains a slightly different algorithm and
produces a plot that looks completely reasonable, which is the worst kind of bug
to have in a week whose entire deliverable is a comparison. So `test_ddqn.py`
targets the specific wrong versions that are easy to write by accident:

- evaluating with the online net instead of the target net
- taking the bootstrap net's own max, which is just the standard target again
- swapping which network selects and which evaluates

Each test is rigged so the wrong implementation produces a *different number*
from the right one, rather than merely a plausible one. There is also a property
test asserting the double target is never above the standard target across 200
random Q tables, which holds by construction since a max over a set is at least
any single element of it, and an identity test asserting the two targets agree
exactly when both networks have the same weights.

`bellman_target()` is a module-level function rather than inline in the training
loop specifically so the tests exercise the real code path instead of
re-deriving it.

## Where this ran

On an Apple Silicon Mac mini via the MPS backend, not on Nautilus. Throughput was
the binding constraint in task 7 and it still is here.
