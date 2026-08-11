"""
PPO from scratch, applied to CarRacing-v3.

This is a direct implementation of Schulman et al. 2017 (arXiv:1707.06347).
No stable-baselines3, no RL library. Only PyTorch, NumPy and Gymnasium.

The pieces of the paper that show up here:
  - the clipped surrogate objective, Equation 7
  - generalized advantage estimation, Equations 11 and 12
  - the combined loss with a value term and an entropy bonus, Equation 9
  - Algorithm 1: collect a fixed-length rollout, then K epochs of minibatch SGD

Run:
    python ppo_carracing.py --steps 1000000 --run-name run1
    python ppo_carracing.py --eval checkpoints/run1_best.pt --episodes 5 --video
"""

import argparse
import csv
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

import gymnasium as gym


# --------------------------------------------------------------------------
# Environment wrapper
# --------------------------------------------------------------------------
class CarRacingWrapper:
    """
    Turns raw CarRacing-v3 into something a small CNN can learn from.

    Three things happen here:

    1. Grayscale + frame stacking. One frame tells you where the car is but not
       how fast it is going or which way it is rotating. Stacking the last 4
       frames puts that motion information into the observation.

    2. Action repeat. Each chosen action is held for several environment steps.
       A single frame of steering barely changes anything, so repeating it makes
       the effect of an action actually visible and cuts the number of network
       forward passes by the same factor.

    3. An early exit. If the car has been earning negative reward for a long
       stretch it is almost certainly parked in the grass. Ending the episode
       there stops the agent from burning thousands of steps on a dead run.
    """

    def __init__(self, seed=0, img_stack=4, action_repeat=8, render_mode=None):
        self.env = gym.make("CarRacing-v3", render_mode=render_mode)
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.img_stack = img_stack
        self.action_repeat = action_repeat
        self.reward_history = deque(maxlen=100)
        self.stack = None

    @staticmethod
    def _to_gray(rgb):
        """RGB uint8 (96, 96, 3) -> float32 (96, 96) scaled to roughly [-1, 1]."""
        gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        return (gray / 128.0 - 1.0).astype(np.float32)

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        self.reward_history.clear()
        gray = self._to_gray(obs)
        # At the start there is no history, so fill the stack with copies.
        self.stack = [gray] * self.img_stack
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False

        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, _ = self.env.step(action)

            # Reaching the end of the track ends the episode with a large bonus
            # already baked into the reward. Do not also treat it as a failure.
            if terminated:
                reward += 100.0

            # A mostly green frame means the car is off the track. A small
            # penalty per off-track step discourages cutting across the grass.
            if np.mean(obs[:, :, 1]) > 185.0:
                reward -= 0.05

            total_reward += reward
            if terminated or truncated:
                break

        # Early exit when the last 100 steps have all been going badly.
        self.reward_history.append(total_reward)
        stuck = (
            len(self.reward_history) == self.reward_history.maxlen
            and np.mean(self.reward_history) <= -0.1
        )

        gray = self._to_gray(obs)
        self.stack.pop(0)
        self.stack.append(gray)

        return np.array(self.stack), total_reward, terminated or stuck, truncated

    def close(self):
        self.env.close()


# --------------------------------------------------------------------------
# Parallel actors
# --------------------------------------------------------------------------
def _worker(remote, parent_remote, seed, img_stack, action_repeat):
    """One child process owning one copy of the environment."""
    parent_remote.close()
    env = CarRacingWrapper(seed=seed, img_stack=img_stack, action_repeat=action_repeat)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, terminated, truncated = env.step(data)
                if terminated or truncated:
                    # Auto-reset so the parent never stalls waiting on one env.
                    # The observation returned is the first of the NEW episode;
                    # `done` tells the learner to cut the bootstrap here.
                    obs = env.reset()
                remote.send((obs, reward, terminated, truncated))
            elif cmd == "reset":
                remote.send(env.reset())
            elif cmd == "close":
                env.close()
                remote.close()
                break
    except KeyboardInterrupt:
        env.close()


class VecCarRacing:
    """
    N environments stepping in parallel, one OS process each.

    This is the "for actor = 1, 2, ..., N" line of Algorithm 1. It matters more
    than it looks: CarRacing spends nearly all its time in Box2D physics and
    rendering the 96x96 observation, which is single-threaded CPU work. One
    environment leaves every other core idle while the GPU waits. N environments
    keep the cores busy and let the network score all N observations in a single
    batched forward pass, which costs barely more than scoring one.
    """

    def __init__(self, num_envs, seed=0, img_stack=4, action_repeat=8):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")  # fork is unsafe with pygame/Box2D on macOS

        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])
        self.processes = []
        for i, (wr, r) in enumerate(zip(self.work_remotes, self.remotes)):
            p = ctx.Process(
                target=_worker,
                args=(wr, r, seed + i * 1000, img_stack, action_repeat),
                daemon=True,
            )
            p.start()
            self.processes.append(p)
            wr.close()

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return np.stack([r.recv() for r in self.remotes])

    def step(self, actions):
        # Send every action first, then collect. Sending and receiving one at a
        # time would serialise the workers and throw away the whole point.
        for r, a in zip(self.remotes, actions):
            r.send(("step", a))
        results = [r.recv() for r in self.remotes]
        obs, rewards, terminated, truncated = zip(*results)
        return (
            np.stack(obs),
            np.array(rewards, dtype=np.float32),
            np.array(terminated, dtype=bool),
            np.array(truncated, dtype=bool),
        )

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.processes:
            p.join(timeout=5)


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """
    One CNN trunk feeding two heads, the actor-critic setup from Section 5.

    The actor outputs the parameters of a Beta distribution rather than a
    Gaussian. CarRacing's three controls are all bounded, and a Beta lives on
    [0, 1] by construction, so no probability mass ever lands on an action the
    environment cannot execute. A Gaussian would need clipping, which quietly
    breaks the ratio in the PPO objective.

    Actions come out as three numbers in [0, 1] and get mapped to:
        steering  ->  a[0] * 2 - 1   (so -1 hard left, +1 hard right)
        gas       ->  a[1]
        brake     ->  a[2]
    """

    def __init__(self, img_stack=4):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(img_stack, 8, kernel_size=4, stride=2), nn.ReLU(),   # 96 -> 47
            nn.Conv2d(8, 16, kernel_size=3, stride=2), nn.ReLU(),          # 47 -> 23
            nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ReLU(),         # 23 -> 11
            nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ReLU(),         # 11 -> 5
            nn.Conv2d(64, 128, kernel_size=3, stride=1), nn.ReLU(),        # 5 -> 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1), nn.ReLU(),       # 3 -> 1
            nn.Flatten(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(256, 100), nn.ReLU(),
            nn.Linear(100, 1),
        )
        self.policy_trunk = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Linear(100, 3)
        self.beta_head = nn.Linear(100, 3)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        z = self.encoder(x)
        value = self.value_head(z)
        h = self.policy_trunk(z)
        # softplus keeps both parameters positive; +1 keeps the distribution
        # unimodal, which stops the policy collapsing onto the extremes early on
        alpha = F.softplus(self.alpha_head(h)) + 1.0
        beta = F.softplus(self.beta_head(h)) + 1.0
        return alpha, beta, value

    # A Beta sample is supposed to be open on (0, 1), but in float32 it can round
    # to exactly 0.0 or 1.0. log_prob there is -inf, which makes the probability
    # ratio inf and then NaN, and training dies with no obvious cause. Nudging
    # samples inside the interval costs nothing and avoids that.
    EPS = 1e-6

    def act(self, x):
        """Sample an action and return it with its log-probability and value."""
        alpha, beta, value = self(x)
        dist = Beta(alpha, beta)
        action = dist.sample().clamp(self.EPS, 1.0 - self.EPS)
        log_prob = dist.log_prob(action).sum(dim=1)
        return action, log_prob, value.squeeze(-1)

    def evaluate(self, x, action):
        """Score a stored action under the *current* policy. Used during updates."""
        alpha, beta, value = self(x)
        dist = Beta(alpha, beta)
        log_prob = dist.log_prob(action.clamp(self.EPS, 1.0 - self.EPS)).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        return log_prob, entropy, value.squeeze(-1)


# --------------------------------------------------------------------------
# Rollout storage
# --------------------------------------------------------------------------
class RolloutBuffer:
    """
    One fixed-length trajectory segment per actor, the length-T buffer in Algorithm 1.

    Everything is shaped (T, N): T timesteps for each of N parallel actors. The
    advantage recursion runs down the time axis independently for every actor,
    then the whole thing flattens to T*N samples for minibatch SGD.
    """

    def __init__(self, size, num_envs, img_stack, device):
        self.size = size
        self.num_envs = num_envs
        self.device = device
        self.obs = np.zeros((size, num_envs, img_stack, 96, 96), dtype=np.float32)
        self.actions = np.zeros((size, num_envs, 3), dtype=np.float32)
        self.log_probs = np.zeros((size, num_envs), dtype=np.float32)
        self.rewards = np.zeros((size, num_envs), dtype=np.float32)
        self.values = np.zeros((size, num_envs), dtype=np.float32)
        self.dones = np.zeros((size, num_envs), dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, done):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = done
        self.ptr += 1

    @property
    def full(self):
        return self.ptr >= self.size

    @property
    def total_samples(self):
        return self.size * self.num_envs

    def compute_gae(self, last_values, gamma, lam):
        """
        Generalized advantage estimation, Equations 11 and 12 of the paper.

        Walking backwards through the segment, for each actor independently:
            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            A_t     = delta_t + gamma * lam * A_{t+1}

        lam trades bias against variance. lam = 1 is the plain discounted return
        (unbiased, noisy); lam = 0 is a one-step TD estimate (biased, smooth).
        0.95 is the paper's value and sits close to the low-variance end.

        `last_values` is one bootstrap value per actor, shape (N,).
        """
        last_values = np.asarray(last_values, dtype=np.float32).reshape(self.num_envs)
        advantages = np.zeros((self.size, self.num_envs), dtype=np.float32)
        running = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.size)):
            next_value = last_values if t == self.size - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            running = delta + gamma * lam * next_non_terminal * running
            advantages[t] = running
        returns = advantages + self.values
        return advantages, returns

    def to_tensors(self, advantages, returns):
        """Flatten (T, N, ...) to (T*N, ...) so minibatches can mix actors."""
        n = self.total_samples
        t = lambda x, shape: torch.as_tensor(
            np.ascontiguousarray(x.reshape(shape)), device=self.device
        )
        return (
            t(self.obs, (n,) + self.obs.shape[2:]),
            t(self.actions, (n, 3)),
            t(self.log_probs, (n,)),
            t(advantages, (n,)),
            t(returns, (n,)),
        )

    def clear(self):
        self.ptr = 0


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------
def save_checkpoint(path, net, optimizer, global_step, episode, best_avg, recent):
    """Everything needed to pick up exactly where training stopped."""
    torch.save({
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "episode": episode,
        "best_avg": best_avg,
        "recent": list(recent),
    }, path)


def load_state_dict(path, device):
    """Accepts both a full checkpoint and a bare state_dict."""
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"], ckpt
    return ckpt, None


def truncate_log(path, max_step, step_col=1):
    """
    Drop log rows recorded after the checkpoint we are resuming from.

    A run that dies between checkpoint saves leaves the CSV ahead of the saved
    weights. Appending from there would log two different updates at the same
    step and quietly corrupt the graphs. Rewinding the log to match the
    checkpoint keeps the two in step.

    Returns the surviving rows, header excluded.
    """
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    kept = [r for r in body if r and int(r[step_col]) <= max_step]
    if len(kept) != len(body):
        print(f"  rewound {os.path.basename(path)}: dropped "
              f"{len(body) - len(kept)} row(s) written after the checkpoint")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)
    return kept


def _last_col(path, col, default):
    """Value in `col` of the last data row, or `default` if there are none."""
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r and r[0].isdigit()]
    return rows[-1][col] if rows else default


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n_envs = max(1, args.num_envs)
    env = VecCarRacing(
        n_envs, seed=args.seed, img_stack=args.img_stack,
        action_repeat=args.action_repeat,
    )
    # rollout is the total batch per update; split it across the actors so the
    # number of samples per update stays the same whatever N is
    steps_per_env = max(1, args.rollout // n_envs)
    print(f"actors: {n_envs} | {steps_per_env} steps each "
          f"| {steps_per_env * n_envs} samples per update")

    net = ActorCritic(args.img_stack).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    buffer = RolloutBuffer(steps_per_env, n_envs, args.img_stack, device)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)

    episode, global_step, update = 0, 0, 0
    recent = deque(maxlen=100)
    best_avg = -np.inf

    # Resuming matters more than it looks. Colab disconnects long runs, and
    # restarting a 4-hour job from zero because the tab went idle is miserable.
    if args.resume:
        state, ckpt = load_state_dict(args.resume, device)
        net.load_state_dict(state)
        if ckpt is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            global_step = ckpt["global_step"]
            episode = ckpt["episode"]
            best_avg = ckpt["best_avg"]
            recent = deque(ckpt["recent"], maxlen=100)
            print(f"resumed from {args.resume} at step {global_step:,}, episode {episode}")
        else:
            print(f"loaded weights only from {args.resume} (no optimizer state)")

    ep_log_path = os.path.join(args.out_dir, f"{args.run_name}_episodes.csv")
    up_log_path = os.path.join(args.out_dir, f"{args.run_name}_updates.csv")

    # Append when resuming so one run's history stays in one file.
    resuming = bool(args.resume) and os.path.exists(ep_log_path)
    if resuming:
        # Rewind both logs to the checkpoint before appending, then take the
        # update counter from what survived rather than from the stale tail.
        truncate_log(ep_log_path, global_step)
        kept = truncate_log(up_log_path, global_step)
        update = int(kept[-1][0]) if kept else 0
        episode = int(_last_col(ep_log_path, 0, episode))

    mode = "a" if resuming else "w"
    ep_file = open(ep_log_path, mode, newline="")
    up_file = open(up_log_path, mode, newline="")
    ep_writer = csv.writer(ep_file)
    up_writer = csv.writer(up_file)
    if mode == "w":
        ep_writer.writerow(["episode", "step", "reward", "length", "running_avg"])
        up_writer.writerow([
            "update", "step", "policy_loss", "value_loss", "entropy",
            "approx_kl", "clip_fraction", "explained_variance", "lr", "blowup_frac",
        ])

    obs = env.reset()
    ep_reward = np.zeros(n_envs, dtype=np.float64)
    ep_len = np.zeros(n_envs, dtype=np.int64)
    start = time.time()
    start_step = global_step   # so steps/s measures THIS session, not the total

    while global_step < args.steps:

        # ---- Phase 1: collect a rollout with the current (frozen) policy ----
        net.eval()
        while not buffer.full:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, log_prob, value = net.act(obs_t)
            a = action.cpu().numpy()                      # (N, 3) in [0, 1]

            env_actions = np.stack(
                [a[:, 0] * 2.0 - 1.0, a[:, 1], a[:, 2]], axis=1
            ).astype(np.float32)
            next_obs, reward, terminated, truncated = env.step(env_actions)
            done = terminated | truncated

            buffer.add(obs, a, log_prob.cpu().numpy(), reward,
                       value.cpu().numpy(), done.astype(np.float32))
            obs = next_obs
            ep_reward += reward
            ep_len += 1
            global_step += n_envs

            for i in np.nonzero(done)[0]:
                episode += 1
                recent.append(float(ep_reward[i]))
                running_avg = float(np.mean(recent))
                ep_writer.writerow([episode, global_step, round(float(ep_reward[i]), 2),
                                    int(ep_len[i]), round(running_avg, 2)])
                if episode % 10 == 0:
                    ep_file.flush()
                    elapsed = time.time() - start
                    print(f"ep {episode:5d} | step {global_step:8d} | "
                          f"reward {ep_reward[i]:8.2f} | avg100 {running_avg:8.2f} | "
                          f"{(global_step-start_step)/max(elapsed,1e-9):6.1f} steps/s | "
                          f"{elapsed/60:.1f} min")
                if len(recent) >= 20 and running_avg > best_avg:
                    best_avg = running_avg
                    save_checkpoint(
                        os.path.join(args.out_dir, "checkpoints", f"{args.run_name}_best.pt"),
                        net, optimizer, global_step, episode, best_avg, recent,
                    )
                # the worker already reset this env; just clear our counters
                ep_reward[i] = 0.0
                ep_len[i] = 0

        # ---- Annealing, the alpha of Table 5 ----
        # The paper scales the Adam stepsize (and the clip range, on Atari) by a
        # factor annealed from 1 to 0 across training. Late in a long run the
        # policy is close to good, and a full-size step is more likely to wreck
        # it than improve it. Shrinking the step size is what stops a run that
        # peaked at hour four from collapsing by hour seven.
        if args.anneal:
            alpha = max(0.0, 1.0 - global_step / args.steps)
            for g in optimizer.param_groups:
                g["lr"] = args.lr * alpha
            clip_now = max(args.clip * alpha, 0.01)
        else:
            alpha = 1.0
            clip_now = args.clip

        # ---- Phase 2: K epochs of minibatch SGD on that rollout ----
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            _, _, last_values = net(obs_t)
            last_values = last_values.squeeze(-1).cpu().numpy()

        advantages, returns = buffer.compute_gae(last_values, args.gamma, args.lam)
        b_obs, b_actions, b_old_log_probs, b_adv, b_returns = buffer.to_tensors(
            advantages, returns
        )

        # Normalizing advantages per batch is not in the paper, but it is standard
        # practice now. It keeps the gradient scale steady as reward magnitudes grow.
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        net.train()
        indices = np.arange(buffer.total_samples)
        kls, clip_fracs, p_losses, v_losses, entropies = [], [], [], [], []
        blowup_fracs = []

        for _ in range(args.epochs):
            np.random.shuffle(indices)
            for start_i in range(0, buffer.total_samples, args.batch_size):
                mb = indices[start_i:start_i + args.batch_size]
                if len(mb) < 2:
                    continue
                mb_t = torch.as_tensor(mb, device=device)

                log_prob, entropy, value = net.evaluate(b_obs[mb_t], b_actions[mb_t])

                # ---- the clipped surrogate objective, Equation 7 ----
                ratio = torch.exp(log_prob - b_old_log_probs[mb_t])
                unclipped = ratio * b_adv[mb_t]
                clipped = torch.clamp(ratio, 1 - clip_now, 1 + clip_now) * b_adv[mb_t]
                # the paper maximizes the min of the two, so we minimize its negative
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = F.smooth_l1_loss(value, b_returns[mb_t])
                entropy_bonus = entropy.mean()

                # Equation 9, with the sign flipped because we are minimizing
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_bonus

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    # Schulman's k3 estimator: mean(r - 1 - log r).
                    # It is unbiased and non-negative, but it grows linearly in r,
                    # so one blown-up ratio drags the mean into the 1e30s and the
                    # diagnostic becomes unreadable. Work in log space and cap the
                    # exponent, then report the median instead of the mean so a
                    # handful of outliers cannot dominate.
                    logr = (log_prob - b_old_log_probs[mb_t]).clamp(-20.0, 20.0)
                    r_safe = logr.exp()
                    approx_kl = (r_safe - 1.0 - logr).median()
                    if not torch.isfinite(approx_kl):
                        approx_kl = torch.tensor(float("nan"))
                    clip_frac = ((ratio - 1).abs() > clip_now).float().mean()
                    # how often the policy moved so far that the stored action is
                    # now essentially impossible. This is the collapse warning sign.
                    blowups = (logr.abs() > 10).float().mean()

                p_losses.append(policy_loss.item())
                v_losses.append(value_loss.item())
                entropies.append(entropy_bonus.item())
                kls.append(approx_kl.item())
                clip_fracs.append(clip_frac.item())
                blowup_fracs.append(blowups.item())

        # how much of the return variance the value head actually explains
        y_true = returns.ravel()
        y_pred = buffer.values.ravel()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        update += 1
        up_writer.writerow([
            update, global_step,
            round(float(np.mean(p_losses)), 5),
            round(float(np.mean(v_losses)), 5),
            round(float(np.mean(entropies)), 5),
            round(float(np.mean(kls)), 6),
            round(float(np.mean(clip_fracs)), 4),
            round(float(explained_var), 4),
            round(args.lr * alpha, 7),
            round(float(np.mean(blowup_fracs)), 5),
        ])
        up_file.flush()
        buffer.clear()

        # Save on the very first update too. Otherwise a run that dies early has
        # nothing at all to resume from.
        if update == 1 or update % args.save_every == 0:
            save_checkpoint(
                os.path.join(args.out_dir, "checkpoints", f"{args.run_name}_latest.pt"),
                net, optimizer, global_step, episode, best_avg, recent,
            )

    save_checkpoint(
        os.path.join(args.out_dir, "checkpoints", f"{args.run_name}_final.pt"),
        net, optimizer, global_step, episode, best_avg, recent,
    )
    ep_file.close()
    up_file.close()
    env.close()
    print(f"\ndone in {(time.time() - start)/60:.1f} min")
    print(f"best 100-episode average: {best_avg:.2f}")
    print(f"logs: {ep_log_path}\n      {up_log_path}")


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ActorCritic(args.img_stack).to(device)
    state, ckpt = load_state_dict(args.eval, device)
    net.load_state_dict(state)
    net.eval()
    if ckpt is not None:
        print(f"checkpoint from step {ckpt['global_step']:,}, "
              f"best 100-episode average {ckpt['best_avg']:.1f}")

    render_mode = "rgb_array" if args.video else None
    env = CarRacingWrapper(
        seed=args.seed + 1000, img_stack=args.img_stack,
        action_repeat=args.action_repeat, render_mode=render_mode,
    )

    if args.video:
        from gymnasium.wrappers import RecordVideo
        env.env = RecordVideo(
            env.env, video_folder=os.path.join(args.out_dir, "videos"),
            episode_trigger=lambda i: True, name_prefix=args.run_name,
        )

    scores = []
    for ep in range(args.episodes):
        obs = env.reset()
        total, done = 0.0, False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                alpha, beta, _ = net(obs_t)
            # At evaluation time use the distribution mean instead of sampling,
            # so the run is deterministic and shows what the policy has learned.
            a = (alpha / (alpha + beta)).cpu().numpy()[0]
            env_action = np.array([a[0] * 2.0 - 1.0, a[1], a[2]], dtype=np.float32)
            obs, reward, terminated, truncated = env.step(env_action)
            total += reward
            done = terminated or truncated
        scores.append(total)
        print(f"episode {ep + 1}: {total:.2f}")

    env.close()
    print(f"\nmean over {args.episodes} episodes: {np.mean(scores):.2f} "
          f"(std {np.std(scores):.2f})")


# --------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="PPO from scratch on CarRacing-v3")

    p.add_argument("--steps", type=int, default=1_000_000,
                   help="total environment steps (after action repeat)")
    p.add_argument("--run-name", type=str, default="run1")
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--seed", type=int, default=0)

    # environment
    p.add_argument("--img-stack", type=int, default=4)
    p.add_argument("--action-repeat", type=int, default=8)

    # PPO hyperparameters. Defaults follow the paper where they transfer, and
    # follow what is known to work on CarRacing where they do not.
    p.add_argument("--num-envs", type=int, default=8,
                   help="N parallel actors, the outer loop of Algorithm 1. Set this "
                        "near your CPU core count; CarRacing is CPU-bound")
    p.add_argument("--rollout", type=int, default=2000,
                   help="T, timesteps collected before each update (paper used 2048)")
    p.add_argument("--epochs", type=int, default=10, help="K, paper used 10")
    p.add_argument("--batch-size", type=int, default=128, help="paper used 64")
    p.add_argument("--clip", type=float, default=0.1,
                   help="epsilon. Paper's best on MuJoCo was 0.2; 0.1 is steadier here")
    p.add_argument("--lr", type=float, default=1e-3, help="paper used 3e-4 on MuJoCo")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--vf-coef", type=float, default=1.0, help="c1 in Equation 9")
    p.add_argument("--ent-coef", type=float, default=0.0,
                   help="c2 in Equation 9. Beta already explores, so 0 is a fine default")
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--no-anneal", dest="anneal", action="store_false",
                   help="disable the linear decay of learning rate and clip range. "
                        "The paper anneals both; leaving this on is recommended for long runs")
    p.set_defaults(anneal=True)
    p.add_argument("--save-every", type=int, default=2,
                   help="updates between checkpoints. The network is under 2 MB, so "
                        "saving often is cheap insurance against a disconnect")
    p.add_argument("--resume", type=str, default=None,
                   help="checkpoint to continue from. --steps is an absolute target, "
                        "so pass a larger number than the run reached")

    # evaluation
    p.add_argument("--eval", type=str, default=None, help="path to a checkpoint")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--video", action="store_true", help="record mp4s during eval")

    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)
