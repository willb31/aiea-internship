# PPO from scratch on CarRacing-v3

Task 6. An implementation of Proximal Policy Optimization built directly from
Schulman et al. 2017 ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347)), with no RL
library involved. The other script in this repo, `car_racing_rl.py`, uses
stable-baselines3. This folder does not.

Only PyTorch, NumPy and Gymnasium are used. Everything specific to PPO is written out
in `ppo_carracing.py`.

## Where the paper shows up in the code

| Paper | Code |
|---|---|
| Clipped surrogate objective, Eq. 7 | `torch.min(unclipped, clipped)` in `train()` |
| GAE, Eq. 11 and 12 | `RolloutBuffer.compute_gae()` |
| Combined loss, Eq. 9 | `policy_loss + vf_coef * value_loss - ent_coef * entropy` |
| Algorithm 1 | the collect / update loop in `train()` |

Two choices differ from the paper, both because CarRacing is not MuJoCo:

**Beta policy instead of Gaussian.** Steering, gas and brake are all bounded. A Beta
distribution lives on [0, 1] by construction, so it never puts probability on an action
the environment would have to clip. Clipping a Gaussian would corrupt the probability
ratio that the whole PPO objective is built on.

**A CNN encoder.** The observation is a 96x96 image, not a state vector, so the policy
and value heads sit on top of a shared six-layer convolutional trunk.

## Parallel actors, and why this is the setting that matters

Algorithm 1 in the paper begins "for actor = 1, 2, ..., N". `--num-envs` is that N, and on
this environment it decides how long your run takes.

CarRacing is CPU-bound, not GPU-bound. Every agent step runs 8 Box2D physics frames and
renders 8 images to build the observation, all single-threaded. The network is under 2 MB,
so the GPU finishes instantly and then waits. Measured on a 4-core machine:

| Actors | Steps/sec |
|---|---|
| 1 | 30 |
| 4 | 57 |
| 6 | 53 (no gain, only 4 cores) |

Set `--num-envs` near your core count and no higher. Past that the processes just fight
over the same cores.

For reference, a free Colab T4 managed 13.7 steps/sec with a single actor, because Colab
gives you 2 vCPUs. A GPU does not rescue you here. Cores do.

`--rollout` is the total number of samples per update, split across the actors, so the
batch size per update stays the same whatever N you pick.

## Environment handling

Three wrappers do most of the work in making this learnable:

- **Frame stacking (4 frames).** A single frame shows position but not velocity or
  rotation. Four stacked grayscale frames carry the motion.
- **Action repeat (8 steps).** One frame of steering barely moves the car. Holding each
  action makes its effect visible and cuts network forward passes by 8x.
- **Early exit.** If the running reward over the last 100 steps is below -0.1, the car is
  sitting in the grass and the episode ends. Without this, bad runs waste thousands of steps.

## Running it on macOS (Apple Silicon)

```bash
brew install swig
cd task6-ppo-from-scratch
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy "gymnasium[box2d]" pygame matplotlib pandas moviepy

python test_ppo.py          # ~5 seconds, run this first

# how many cores do you have?
sysctl -n hw.ncpu

# train, setting --num-envs to about your core count
python ppo_carracing.py --steps 1000000 --num-envs 8 --run-name run1
```

If `brew install swig` is not an option, `pip install swig` usually works too, and the
build-isolation workaround below covers the case where it does not.

Apple Silicon runs the network on the MPS backend automatically. That barely matters,
because the bottleneck is the CPU-side simulation, which is exactly why `--num-envs` is
the setting worth tuning.

## Running it

```bash
pip install -r requirements.txt

# train
python ppo_carracing.py --steps 1000000 --run-name run1

# graphs
python plot_results.py --run-name run1

# watch it drive, and record video
python ppo_carracing.py --eval results/checkpoints/run1_best.pt --episodes 5 --video
```

`train_colab.ipynb` runs the whole thing on a free Colab GPU if you do not have one locally.

Box2D needs SWIG installed first. On most systems:

```bash
pip install swig
pip install "gymnasium[box2d]"
```

If `box2d-py` fails to build with `ModuleNotFoundError: No module named 'swig'`, pip's build
isolation is hiding the SWIG you just installed from the build. This works around it:

```bash
PYTHONPATH=$(python -c "import site; print(site.getsitepackages()[0])") \
  pip install --no-build-isolation box2d-py==2.3.5
pip install gymnasium pygame
```

## Hyperparameters

Defaults follow the paper where they transfer and follow what works on CarRacing where
they do not. All of them are command-line flags.

| Setting | Value | Paper |
|---|---|---|
| Rollout length T | 2000 total | 2048 per actor |
| Parallel actors N | 8 | not stated (Atari used 8) |
| Epochs K | 10 | 10 |
| Minibatch size | 128 | 64 |
| Clip epsilon | 0.1 | 0.2 was best on MuJoCo |
| Learning rate | 1e-3 | 3e-4 |
| Discount gamma | 0.99 | 0.99 |
| GAE lambda | 0.95 | 0.95 |
| Value coefficient c1 | 1.0 | not stated for MuJoCo |
| Entropy coefficient c2 | 0.0 | 0.01 on Atari |
| LR / clip annealing | on | alpha annealed 1 to 0 (Table 5) |

The entropy bonus is off by default because a Beta policy already keeps enough spread
early in training. It is still logged so you can see exploration decay on its own.

Clip epsilon is 0.1 rather than the paper's 0.2. Image observations produce noisier
advantage estimates than MuJoCo state vectors, and the tighter window keeps updates from
overreacting to that noise.

## Annealing

Both the learning rate and the clip range decay linearly to zero across `--steps`, which is
the alpha factor of the paper's Table 5. Late in training the policy is already close to
good and a full-size update is more likely to break it than improve it. Disable with
`--no-anneal` if you want to see the difference.

Because the schedule is tied to `--steps`, set that to the number you actually intend to
finish. Stopping a 3M-step run at 1M leaves the rate still high.

## Resuming

Colab disconnects long runs. `--resume` picks up from a checkpoint with the optimizer
state, step count and episode history intact:

```bash
python ppo_carracing.py --steps 1000000 --run-name run1 \
    --resume results/checkpoints/run1_latest.pt
```

`--steps` is an absolute target, not an increment, so pass a number larger than the run
already reached. If the process died between checkpoint saves, the CSV logs will be ahead
of the saved weights. On resume the logs get rewound to match the checkpoint and the
dropped rows are reported, so the graphs never contain two different updates at the same step.

## Output

Training writes to `results/`:

- `run1_episodes.csv` — reward, length and running average per episode
- `run1_updates.csv` — policy loss, value loss, entropy, approximate KL, clip fraction, explained variance per update
- `checkpoints/` — best, latest and final weights
- `figures/` — the plots, once `plot_results.py` has run

## Reading the graphs

**Learning curve.** Reward per episode with a 100-episode average over it. CarRacing counts
as solved at an average of 900. Expect a long flat stretch near zero first while the agent
learns that gas is worth using at all.

**Clip fraction.** The share of probability ratios that landed outside [1-eps, 1+eps] and
got clipped. This is the paper's mechanism made visible. A healthy run sits somewhere
around 0.05 to 0.2. Near zero means the updates are too timid to matter; very high means
the policy is trying to move much further than the trust region allows.

**Approximate KL.** How far the policy moved per update. It should stay small and stable.
Spikes usually come right before the reward curve falls apart.

**Entropy.** This is differential entropy, so negative numbers are normal and not a bug.
A Beta(1,1) is uniform on [0,1] and has entropy exactly 0. As the policy commits to
particular actions the distribution narrows and entropy goes below zero. What matters is
the trend, not the sign: it should drift downward as the agent stops exploring. If it
plunges early the policy has collapsed onto one action and stopped learning.

**Explained variance.** How much of the return variance the value head accounts for.
Starts near zero or negative and should climb. If it stays near zero, the advantages are
mostly noise and the policy has nothing reliable to learn from.

## Reference

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O.
*Proximal Policy Optimization Algorithms.* arXiv:1707.06347, 2017.
