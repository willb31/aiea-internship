# Task 7: DQN from scratch on a discretized CarRacing-v3

An implementation of Mnih et al. 2013, [Playing Atari with Deep Reinforcement
Learning](https://arxiv.org/abs/1312.5602), written from the paper. No
stable-baselines3 and no other RL library, just PyTorch, NumPy and Gymnasium.

## The experiment

The linked paper is the 2013 NeurIPS workshop version, which has **no target
network**. That came later, in the 2015 Nature paper. So this week runs both:

- **Run A**, `dqn2013`, the paper exactly as published. The bootstrap target
  `max_a' Q(s', a')` comes from the same weights that are being updated, so the
  target moves the moment the network does.
- **Run B**, `dqn2015`, identical seed and hyperparameters with a target network
  synced every 1,000 steps.

Everything else is held constant, so the difference between the two curves is
attributable to that one change.

## Running it

```bash
pip install swig
pip install -r requirements.txt
bash run_task7.sh
```

That runs the tests, both training runs, an evaluation of each best checkpoint on
unseen tracks, and the figures. Each run is capped at 2.5 hours of wall clock.

Individually:

```bash
python test_dqn.py                                       # ~30 seconds, run this first
python dqn_carracing.py --run-name dqn2013 --steps 400000
python dqn_carracing.py --run-name dqn2015 --steps 400000 --target-net
python dqn_carracing.py --eval results/checkpoints/dqn2013_best.pt --episodes 5
python plot_results.py --compare-ppo ../task6-ppo-from-scratch/results/run1_episodes.csv
```

## Why the environment had to change

DQN puts one output unit per action and takes a max over them, so the action
space has to be finite. CarRacing's is not: steering, gas and brake are all
continuous, which is exactly why last week's PPO used a Beta distribution. The
wrapper here exposes five actions: coast, hard left, hard right, accelerate,
brake.

The visual pipeline and reward shaping are otherwise identical to task 6
(grayscale, 4-frame stack, the +100 completion bonus, the off-track penalty, the
stuck-episode cutoff), so the episode returns are comparable between the two
weeks. The one difference is action repeat, 4 here against 8 last week, because
the paper's epsilon schedule is defined per agent decision.

## Two deliberate deviations from the paper

**No reward clipping.** The paper clips every reward to {-1, 0, +1} so that one
hyperparameter set works across seven Atari games with different scoring scales.
CarRacing pays -0.1 per frame and +1000/N per track tile. Clipping would give
"visited a tile" and "idled for a frame" the same magnitude and destroy the only
signal separating driving from sitting still.

**Huber loss instead of squared error.** Squared error on unclipped rewards
produces very large gradients on the occasional big return. This is the same
substitution the 2015 paper makes.

## Files

| file | what it is |
| --- | --- |
| `env_and_replay.py` | discrete action set, CarRacing wrapper, replay memory, epsilon schedule. No torch, so it is testable on its own. |
| `dqn_carracing.py` | Q-network, training loop (Algorithm 1), evaluation, CLI. |
| `test_dqn.py` | correctness checks. Run before any long run. |
| `plot_results.py` | figures, plus it prints every number the write-up needs. |
| `run_task7.sh` | the whole week end to end. |

## A note on the replay buffer

Storing stacked states directly would need 4 frames for the state and 4 for the
next state, 73 KB per transition at 96x96, so a 100k buffer would want 7 GB.
Instead each frame is stored once as uint8 and the stacks are rebuilt by index at
sample time, which costs about 0.9 GB for the same buffer.

The part of that worth testing is episode boundaries: frame `t-1` may belong to
the previous episode, and stacking it into state `t` would hand the network a
state that never existed. `test_dqn.py` covers that case, the ring-wraparound
case, and the distinction between a true terminal (cut the bootstrap) and the
stuck-timeout (do not).
