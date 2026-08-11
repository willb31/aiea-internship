# Task 9: a one-click RL benchmarking framework

Everything from weeks 6 through 8 in a single file, plus one new algorithm, so
that all five can be trained and compared with one command.

```bash
pip install swig
pip install -r requirements.txt

python test_framework.py                    # ~40 seconds, run this first
python rl_benchmark.py --all                # the benchmark, then the figures
```

That trains all five algorithms, writes their logs, draws the figures and prints
the results table. Nothing else to run.

## The five algorithms

| name | what it is |
| --- | --- |
| `ppo` | task 6, clipped objective and GAE, ported to discrete actions |
| `dqn2013` | task 7, Mnih 2013 as published, no target network |
| `dqn2015` | task 7, with the Nature 2015 target network |
| `ddqn` | task 8, Double DQN target |
| `dueling_ddqn` | new this week, value and advantage streams |

Adding a sixth means adding one entry to the `ALGORITHMS` dict. Nothing else in
the file knows how many there are.

## Why PPO had to change, and why the task 6 number will not match

Task 6's PPO used a Beta distribution over the three continuous controls with an
action repeat of 8. Task 7's DQN used 5 discrete actions with an action repeat of
4. Task 7 described the returns as "comparable between the two weeks". They were
close, but they were not measured on the same environment, so strictly they were
not comparable at all, and I have been quoting PPO's 111.2 against DQN's numbers
for two weeks on that basis.

A benchmark cannot hedge like that. PPO here has a Categorical head over the same
5 actions and runs through the identical wrapper, so the comparison is real. The
cost is that the PPO figure this week is not the same experiment as task 6's.

## Counting steps fairly

On-policy and off-policy algorithms want different amounts of parallelism. PPO
does badly with a single environment because its updates need a batch of fresh
trajectories; DQN replays old ones and does not care.

So each agent declares `num_envs`, and the budget is counted in **total
environment transitions**, not iterations of the runner loop. PPO running 4
environments does not get 4 times the data, it gets the same data spread across
4 environments. `test_framework.py` asserts this directly, because a runner that
counted iterations would quietly hand PPO four times the experience and void the
entire comparison.

## Running it other ways

```bash
python rl_benchmark.py --algos ddqn dueling_ddqn      # a subset
python rl_benchmark.py --all --parallel 2             # two at a time
python rl_benchmark.py --sweep lr                     # the learning-rate study
python rl_benchmark.py --plot-only                    # redraw from existing CSVs
python rl_benchmark.py --k8s <image>                  # write Nautilus manifests
```

`--parallel` forks worker processes. On one machine 2 is about the ceiling: the
runs contend for a single GPU, and each DQN replay buffer is roughly 0.9 GB.

## Nautilus

The task asks for the algorithms to be benchmarked *at the same time*, which on a
cluster means one Job per algorithm rather than a process pool. `--k8s` writes
those manifests, one per algorithm, each requesting a GPU and mounting a shared
volume for results. The `Dockerfile` builds the image they run.

Both are **untested**. I have no cluster access to submit them to, so they are a
starting point rather than something known to work. The runs behind this week's
results were sequential on an Apple Silicon Mac mini.

## Output

Per algorithm, in `results/`:

| file | contents |
| --- | --- |
| `{algo}_episodes.csv` | one row per finished episode |
| `{algo}_updates.csv` | loss, value estimates and throughput every 1,000 steps |
| `{algo}_policy.csv` | action distribution over a frozen state set |
| `{algo}_summary.json` | final numbers and the full config used |

Figures land in `results/figures/`: rewards (running average and cumulative),
loss, policies, and value estimates.

The policy log is the one worth explaining. Every algorithm is scored on the same
frozen set of states, collected under the random policy before learning starts,
so the action distributions are directly comparable. For PPO that is the softmax;
for the DQNs, which have no distribution of their own, it is the share of states
where each action is the argmax. Both answer the same question, which is what the
policy actually does rather than what it is worth.

## A note on what the tests are for

A benchmark fails differently from a training script. If two registry entries end
up with identical flags, or one algorithm gets a larger step budget than another,
nothing crashes. You get a clean table of numbers that are wrong, and every
conclusion drawn from it is wrong too.

So `test_framework.py` spends most of its effort there: that the five entries are
five genuinely different algorithms, that the DQN variants differ only in the
three documented flags, that a mistyped config key raises instead of silently
falling back to a default, and that every agent stops at the same step count.
