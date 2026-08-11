"""
Task 9: a single-file benchmarking framework for the RL algorithms from weeks
6 through 8, plus one new one.

    python rl_benchmark.py --all                 # the whole benchmark, then figures
    python rl_benchmark.py --all --parallel 2    # two algorithms at a time
    python rl_benchmark.py --sweep lr            # the learning-rate study
    python rl_benchmark.py --plot-only           # redraw figures from existing CSVs

Everything is in this one file on purpose. The task asks for a one-click
solution, and a benchmark that needs you to remember which of four modules to
import is not one.

The five algorithms
-------------------
    ppo             task 6, on-policy, clipped objective and GAE
    dqn2013         task 7, Mnih 2013 as published, no target network
    dqn2015         task 7, with the Nature 2015 target network
    ddqn            task 8, Double DQN target
    dueling_ddqn    new this week, value and advantage streams

Why PPO had to change
---------------------
Task 6's PPO used a Beta distribution over the three continuous controls, and
task 7's DQN used 5 discrete actions with a different action repeat, 8 against
4. Task 7 called the returns "comparable". They were close, but they were not
measured on the same environment, so strictly they were not comparable at all.

A benchmark cannot hedge like that, so PPO here has a Categorical head over the
identical 5 actions and runs through the identical wrapper. That makes every
number in the results table directly comparable, and it means the PPO figure
quoted this week will not match the one from task 6. That is the point.

Counting steps fairly
---------------------
On-policy and off-policy algorithms want different amounts of parallelism. PPO
does badly with a single environment because its updates need a batch of fresh
trajectories, while DQN replays old ones and does not care.

So each algorithm declares how many environments it wants, and the budget is
counted in *total environment transitions* rather than iterations of the runner
loop. Every algorithm sees exactly `--steps` transitions of experience no matter
how it chose to collect them, which is the usual sample-efficiency convention
and is the only comparison that means anything here.
"""

import argparse
import csv
import json
import os
import random
import time
from collections import deque

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# --------------------------------------------------------------------------
# Action space
# --------------------------------------------------------------------------

# DQN learns Q(s, a) with one output unit per action, so the action space has to
# be finite. CarRacing's is not: steering, gas and brake are all continuous. This
# is the reason last week's PPO used a Beta distribution and this week cannot.
#
# Five actions is the standard minimal set for this environment. Steering is
# full-lock because a discrete agent has no way to express "a little left", and
# gas is 0.8 rather than 1.0 because full throttle in this car mostly produces
# wheelspin.
DISCRETE_ACTIONS = np.array(
    [
        [0.0, 0.0, 0.0],   # 0: coast
        [-1.0, 0.0, 0.0],  # 1: hard left
        [1.0, 0.0, 0.0],   # 2: hard right
        [0.0, 0.8, 0.0],   # 3: accelerate
        [0.0, 0.0, 0.8],   # 4: brake
    ],
    dtype=np.float32,
)


# --------------------------------------------------------------------------
# Environment wrapper
# --------------------------------------------------------------------------
class CarRacingDiscrete:
    """
    CarRacing-v3 with a 5-action discrete interface and the same visual
    pipeline and reward shaping as the task 6 PPO run.

    Keeping the shaping identical matters: the DQN and PPO episode returns are
    only comparable if the environment pays out the same way for both. The one
    thing that differs is action repeat, 4 here against 8 last week, because the
    paper's epsilon schedule is defined per agent decision and 4 keeps this
    closer to the published setup.

    Frames are kept as uint8 rather than the float32 the PPO code used. A float32
    replay buffer of any useful size does not fit in memory; see ReplayBuffer.
    """

    def __init__(self, seed=0, img_stack=4, action_repeat=4, render_mode=None):
        self.env = gym.make("CarRacing-v3", render_mode=render_mode)
        self.env.reset(seed=seed)
        self.img_stack = img_stack
        self.action_repeat = action_repeat
        self.reward_history = deque(maxlen=100)
        self.stack = None

    @staticmethod
    def _to_gray(rgb):
        """RGB uint8 (96, 96, 3) -> uint8 (96, 96). Scaling happens at sample time."""
        gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        return gray.astype(np.uint8)

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        self.reward_history.clear()
        gray = self._to_gray(obs)
        # No history at the start of an episode, so fill the stack with copies.
        self.stack = [gray] * self.img_stack
        return gray, np.array(self.stack)

    def step(self, action_idx):
        action = DISCRETE_ACTIONS[action_idx]
        total_reward = 0.0
        terminated = truncated = False

        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, _ = self.env.step(action)

            # Finishing the track is a success, not a failure. Same bonus as task 6.
            if terminated:
                reward += 100.0

            # A mostly green frame means the car is in the grass.
            if np.mean(obs[:, :, 1]) > 185.0:
                reward -= 0.05

            total_reward += reward
            if terminated or truncated:
                break

        # Cut the episode when the last 100 decisions have all gone badly, so the
        # agent does not spend thousands of steps parked in the grass. Note this
        # is a *time limit* style ending, not a real terminal state, so it is
        # reported separately: bootstrapping through it is correct, and treating
        # it as terminal would teach the agent that being stuck ends the world.
        self.reward_history.append(total_reward)
        stuck = (
            len(self.reward_history) == self.reward_history.maxlen
            and np.mean(self.reward_history) <= -0.1
        )

        gray = self._to_gray(obs)
        self.stack.pop(0)
        self.stack.append(gray)

        return gray, np.array(self.stack), total_reward, terminated, truncated or stuck

    def close(self):
        self.env.close()


# --------------------------------------------------------------------------
# Replay memory
# --------------------------------------------------------------------------
class ReplayBuffer:
    """
    The replay memory D from Section 4, holding the last N transitions.

    Storing whole stacked states would mean 4 frames per transition for the state
    and 4 more for the next state. At 96x96 that is 73 KB per transition, so a
    100k buffer would need 7 GB. Instead this stores each frame exactly once and
    rebuilds the stacks by index at sample time, which costs 9 KB per transition
    and brings the same buffer down to about 0.9 GB.

    The subtlety that makes this worth testing is episode boundaries. Frame t-1
    may belong to the previous episode, in which case stacking it into state t
    would hand the network a state that never existed. _stack walks backwards and
    repeats the oldest valid frame instead of crossing a boundary, which is the
    same thing reset() does at the start of an episode.
    """

    def __init__(self, capacity, img_stack=4, height=96, width=96):
        self.capacity = capacity
        self.img_stack = img_stack
        self.frames = np.zeros((capacity, height, width), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        # terminal = a real end state, bootstrap must be cut
        self.terminals = np.zeros(capacity, dtype=bool)
        # boundary = any episode end, real or a time limit; do not stack across it
        self.boundaries = np.zeros(capacity, dtype=bool)
        self.n_added = 0

    def __len__(self):
        return min(self.n_added, self.capacity)

    @property
    def _oldest(self):
        """Oldest absolute index still present in the ring."""
        return max(0, self.n_added - self.capacity)

    def add(self, frame, action, reward, terminal, boundary):
        i = self.n_added % self.capacity
        self.frames[i] = frame
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminals[i] = terminal
        self.boundaries[i] = boundary
        self.n_added += 1

    def _stack(self, idx):
        """Build the stacked state whose most recent frame is at absolute index idx."""
        out = np.empty((self.img_stack, *self.frames.shape[1:]), dtype=np.uint8)
        out[-1] = self.frames[idx % self.capacity]
        j = idx
        for s in range(self.img_stack - 2, -1, -1):
            prev = j - 1
            # Stop at the start of the ring, or at an episode boundary. Repeating
            # the oldest valid frame mirrors what reset() does.
            if prev < self._oldest or self.boundaries[prev % self.capacity]:
                out[s] = out[s + 1]
            else:
                j = prev
                out[s] = self.frames[j % self.capacity]
        return out

    def sample(self, batch_size, rng):
        """
        Uniform sample, as in the paper. Valid indices need one stored successor
        (for the next state) and are drawn from the part of the ring that has not
        been overwritten.
        """
        lo = self._oldest
        hi = self.n_added - 2  # need idx + 1 to exist
        if hi < lo:
            raise ValueError("replay buffer does not hold a full transition yet")

        idxs = rng.integers(lo, hi + 1, size=batch_size)

        states = np.empty((batch_size, self.img_stack, *self.frames.shape[1:]), dtype=np.uint8)
        next_states = np.empty_like(states)
        for k, idx in enumerate(idxs):
            states[k] = self._stack(idx)
            next_states[k] = self._stack(idx + 1)

        ring = idxs % self.capacity
        return (
            states,
            self.actions[ring],
            self.rewards[ring],
            next_states,
            self.terminals[ring],
        )


# --------------------------------------------------------------------------
# Exploration schedule
# --------------------------------------------------------------------------
def epsilon_at(step, start, end, decay_steps):
    """Linear anneal from `start` to `end` over `decay_steps`, then flat."""
    if step >= decay_steps:
        return end
    return start + (end - start) * (step / decay_steps)

# ==========================================================================
# Networks
# ==========================================================================
class QNetwork(nn.Module):
    """
    The architecture from Mnih 2013 Section 4.1, with the input left at
    CarRacing's native 96x96 so no rescaling step is needed.

        conv 16 filters, 8x8, stride 4   ->  23x23
        conv 32 filters, 4x4, stride 2   ->  10x10
        fully connected, 256 units
        fully connected, one output per action
    """

    def __init__(self, img_stack=4, n_actions=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(img_stack, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 10 * 10, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        x = x.float() / 128.0 - 1.0
        return self.head(self.features(x))


class DuelingQNetwork(nn.Module):
    """
    Wang et al. 2016, arXiv:1511.06581. Same convolutions, but the head splits
    into a scalar V(s) and a per-action advantage A(s, a), recombined as

        Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')

    The subtraction is the part that matters. Without it the decomposition is
    unidentifiable: adding a constant to V and taking the same constant off every
    A leaves Q unchanged, so nothing pins down which stream learns what and the
    two drift. Forcing the advantages to average zero fixes that.

    The reason to want this on CarRacing is that most states on a straight are
    ones where the action barely matters. A single-stream network has to learn
    the value of that state five separate times, once per action. Splitting the
    streams lets it learn the state's value once.
    """

    def __init__(self, img_stack=4, n_actions=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(img_stack, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        self.value = nn.Sequential(
            nn.Linear(32 * 10 * 10, 256), nn.ReLU(), nn.Linear(256, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(32 * 10 * 10, 256), nn.ReLU(), nn.Linear(256, n_actions),
        )

    def forward(self, x):
        h = self.features(x.float() / 128.0 - 1.0)
        v = self.value(h)
        a = self.advantage(h)
        return v + a - a.mean(dim=1, keepdim=True)


class ActorCritic(nn.Module):
    """
    PPO's network. Same convolutional trunk as the Q networks so the comparison
    is about the algorithm rather than about who got the bigger encoder, then a
    categorical policy head and a scalar value head.

    Task 6 used a Beta distribution over three continuous controls. Here the head
    is a softmax over the same 5 discrete actions the DQN family uses, which is
    what makes the two directly comparable.
    """

    def __init__(self, img_stack=4, n_actions=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(img_stack, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, 256), nn.ReLU(),
        )
        self.policy = nn.Linear(256, n_actions)
        self.value = nn.Linear(256, 1)
        # A small policy-head gain keeps the initial distribution near uniform.
        # Starting with a confident random policy wastes the early rollouts.
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.constant_(self.policy.bias, 0.0)

    def forward(self, x):
        h = self.features(x.float() / 128.0 - 1.0)
        return self.policy(h), self.value(h).squeeze(-1)

    def act(self, x):
        logits, value = self(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, x, action):
        logits, value = self(x)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), value


# ==========================================================================
# Agents
# ==========================================================================
class Agent:
    """
    What the runner needs from an algorithm, and nothing else.

    The runner does not know whether it is driving something on-policy or
    off-policy. It steps environments, hands over what happened, and asks
    whether there are any metrics to log. Everything about how experience is
    stored and when gradients are taken lives behind this interface.

        num_envs      how many environments to run in lockstep
        act           choose actions for a batch of states
        observe       receive the outcome of those actions
        learn         maybe take a gradient step, return metrics or None
        policy_probs  action distribution, for the policy figures
    """

    num_envs = 1

    def act(self, states, step):
        raise NotImplementedError

    def observe(self, frames, actions, rewards, terminated, truncated, next_states):
        raise NotImplementedError

    def learn(self, step):
        return None

    def policy_probs(self, states):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError


class DQNAgent(Agent):
    """
    The whole DQN family. Which one you get depends on three flags:

        target_net   False -> Mnih 2013 as published
                     True  -> the Nature 2015 target network
        double       True  -> Double DQN, online net selects, target net values
        dueling      True  -> the value/advantage head

    They compose, so dueling_ddqn is all three at once. Keeping them as flags on
    one class rather than four subclasses is deliberate: it makes it obvious in
    the diff that the four DQN variants really do share every line that is not
    one of these three branches.
    """

    num_envs = 1

    def __init__(self, cfg, device, n_actions=5, rng=None):
        self.cfg = cfg
        self.device = device
        self.n_actions = n_actions
        self.rng = rng if rng is not None else np.random.default_rng(cfg.seed)

        net = DuelingQNetwork if cfg.dueling else QNetwork
        self.online = net(cfg.img_stack, n_actions).to(device)
        self.target = None
        if cfg.target_net:
            self.target = net(cfg.img_stack, n_actions).to(device)
            self.target.load_state_dict(self.online.state_dict())
            self.target.eval()

        # RMSProp is what the paper uses. Adam would probably do better, but the
        # baseline should be the published one.
        self.opt = torch.optim.RMSprop(
            self.online.parameters(), lr=cfg.lr, alpha=0.95, eps=0.01
        )
        self.buffer = ReplayBuffer(cfg.buffer_size, cfg.img_stack)
        self.eps = cfg.eps_start
        self._pending = None

    def act(self, states, step):
        self.eps = epsilon_at(step, self.cfg.eps_start, self.cfg.eps_end,
                              self.cfg.eps_decay_steps)
        if step < self.cfg.learn_start or random.random() < self.eps:
            return np.array([random.randrange(self.n_actions)])
        with torch.no_grad():
            s = torch.as_tensor(np.asarray(states), device=self.device)
            return self.online(s).argmax(dim=1).cpu().numpy()

    def observe(self, frames, actions, rewards, terminated, truncated, next_states):
        # One env, so index 0 throughout.
        self.buffer.add(
            frames[0], int(actions[0]), float(rewards[0]),
            terminal=bool(terminated[0]),
            boundary=bool(terminated[0] or truncated[0]),
        )

    def learn(self, step):
        if step < self.cfg.learn_start or step % self.cfg.train_freq != 0:
            return None
        if len(self.buffer) < self.cfg.batch_size + 2:
            return None

        states, actions, rewards, next_states, terminals = self.buffer.sample(
            self.cfg.batch_size, self.rng
        )
        states_t = torch.as_tensor(states, device=self.device)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        rewards_t = torch.as_tensor(rewards, device=self.device)
        terminals_t = torch.as_tensor(terminals, device=self.device, dtype=torch.float32)

        q = self.online(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        y = bellman_target(
            self.online, self.target if self.target is not None else self.online,
            next_states_t, rewards_t, terminals_t, self.cfg.gamma, self.cfg.double,
        )

        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.max_grad_norm)
        self.opt.step()

        if self.target is not None and step % self.cfg.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())

        return {
            "loss": float(loss.item()),
            "mean_q": float(q.mean().item()),
            "mean_y": float(y.mean().item()),
            "epsilon": self.eps,
        }

    def policy_probs(self, states):
        """
        A greedy Q policy has no distribution of its own, so this reports the
        share of states for which each action is the argmax. That is the thing
        worth comparing against PPO's softmax: what the policy actually does.
        """
        with torch.no_grad():
            s = torch.as_tensor(np.asarray(states), device=self.device)
            best = self.online(s).argmax(dim=1).cpu().numpy()
        return np.bincount(best, minlength=self.n_actions) / len(best)

    def state_dict(self):
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict() if self.target else None,
            "optimizer": self.opt.state_dict(),
        }


def bellman_target(online, bootstrap, next_states, rewards, terminals, gamma, double):
    """
    The bootstrap target.

        standard:  y = r + gamma * max_a' Q_boot(s', a')
        double:    y = r + gamma * Q_boot(s', argmax_a' Q_online(s', a'))

    `terminals` is 1.0 only for a real terminal. The stuck-episode cutoff is a
    truncation and must keep its bootstrap, because the value of that state is
    not zero, the wrapper just stopped watching.

    Carried over from task 8 unchanged, including its tests.
    """
    with torch.no_grad():
        if double:
            next_actions = online(next_states).argmax(dim=1)
            next_q = bootstrap(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            next_q = bootstrap(next_states).max(dim=1).values
        return rewards + gamma * next_q * (1.0 - terminals)


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """
    Generalised advantage estimation, Schulman et al. 2015.

    Walks backwards accumulating the TD residual discounted by gamma * lambda,
    and cuts the chain wherever an episode really ended. `dones` is 1.0 only for
    a true terminal, so a truncated episode keeps bootstrapping, same rule the
    DQN target uses.

    Pulled out of the training loop so the tests can check it against a hand
    computation. It is the easiest thing in PPO to get subtly wrong, and a wrong
    advantage does not crash, it just trains something else.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        adv[t] = gae
    return adv


class PPOAgent(Agent):
    """
    PPO with the clipped surrogate objective and GAE, from task 6, with the Beta
    head swapped for a Categorical one.

    Runs several environments in lockstep because on-policy updates need a batch
    of fresh trajectories. The budget is still counted in total transitions, so
    this buys no extra data, only less correlated data.
    """

    def __init__(self, cfg, device, n_actions=5, rng=None):
        self.cfg = cfg
        self.device = device
        self.n_actions = n_actions
        self.num_envs = cfg.num_envs

        self.net = ActorCritic(cfg.img_stack, n_actions).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)

        self.rollout = cfg.rollout
        self._reset_storage()
        self._last = None      # (log_prob, value) for the action just taken

    def _reset_storage(self):
        self.s_buf, self.a_buf, self.r_buf = [], [], []
        self.lp_buf, self.v_buf, self.d_buf = [], [], []

    def act(self, states, step):
        with torch.no_grad():
            s = torch.as_tensor(np.asarray(states), device=self.device)
            action, log_prob, value = self.net.act(s)
        self._last = (
            np.asarray(states),
            action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy(),
        )
        return action.cpu().numpy()

    def observe(self, frames, actions, rewards, terminated, truncated, next_states):
        states, acts, log_probs, values = self._last
        self.s_buf.append(states)
        self.a_buf.append(acts)
        self.lp_buf.append(log_probs)
        self.v_buf.append(values)
        self.r_buf.append(np.asarray(rewards, dtype=np.float32))
        # Only a real terminal cuts the bootstrap. A truncation ends the episode
        # but the value of the final state is not zero, same rule as the DQNs.
        self.d_buf.append(np.asarray(terminated, dtype=np.float32))
        self._next_states = np.asarray(next_states)

    def learn(self, step):
        if len(self.r_buf) < self.rollout:
            return None

        cfg = self.cfg
        dev = self.device

        states = torch.as_tensor(np.array(self.s_buf), device=dev)          # T, N, C, H, W
        actions = torch.as_tensor(np.array(self.a_buf), device=dev)
        old_lp = torch.as_tensor(np.array(self.lp_buf), device=dev)
        values = torch.as_tensor(np.array(self.v_buf), device=dev)
        rewards = torch.as_tensor(np.array(self.r_buf), device=dev)
        dones = torch.as_tensor(np.array(self.d_buf), device=dev)

        with torch.no_grad():
            nxt = torch.as_tensor(self._next_states, device=dev)
            _, last_value = self.net(nxt)

        adv = compute_gae(rewards, values, dones, last_value, cfg.gamma, cfg.lam)
        returns = adv + values

        # Flatten time and env into one batch of independent samples.
        b_states = states.reshape(-1, *states.shape[2:])
        b_actions = actions.reshape(-1)
        b_old_lp = old_lp.reshape(-1)
        b_adv = adv.reshape(-1)
        b_returns = returns.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        n = b_states.shape[0]
        idx = np.arange(n)
        losses, clipfracs, entropies = [], [], []

        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, cfg.batch_size):
                mb = idx[start:start + cfg.batch_size]
                mb_t = torch.as_tensor(mb, device=dev)

                lp, entropy, value = self.net.evaluate(b_states[mb_t], b_actions[mb_t])
                ratio = (lp - b_old_lp[mb_t]).exp()

                # The clipped surrogate, Equation 7. The min of the raw and the
                # clipped objective means an update is only allowed to help by so
                # much, but is fully penalised when it hurts.
                unclipped = ratio * b_adv[mb_t]
                clipped = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * b_adv[mb_t]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(value, b_returns[mb_t])
                entropy_loss = -entropy.mean()

                loss = policy_loss + cfg.vf_coef * value_loss + cfg.ent_coef * entropy_loss

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                losses.append(float(loss.item()))
                entropies.append(float(entropy.mean().item()))
                clipfracs.append(float(((ratio - 1).abs() > cfg.clip).float().mean().item()))

        self._reset_storage()
        return {
            "loss": float(np.mean(losses)),
            "mean_q": float(values.mean().item()),   # value estimate, same column
            "mean_y": float(returns.mean().item()),
            "entropy": float(np.mean(entropies)),
            "clip_frac": float(np.mean(clipfracs)),
            "epsilon": 0.0,
        }

    def policy_probs(self, states):
        with torch.no_grad():
            s = torch.as_tensor(np.asarray(states), device=self.device)
            logits, _ = self.net(s)
            return F.softmax(logits, dim=1).mean(dim=0).cpu().numpy()

    def state_dict(self):
        return {"net": self.net.state_dict(), "optimizer": self.opt.state_dict()}


# ==========================================================================
# Configuration and the algorithm registry
# ==========================================================================
class Config:
    """Plain attribute bag. Defaults are the shared ones; the registry overrides."""

    def __init__(self, **kw):
        # shared
        self.steps = 200_000
        self.seed = 0
        self.img_stack = 4
        self.action_repeat = 4
        self.gamma = 0.99
        self.max_grad_norm = 10.0
        self.lr = 2.5e-4
        self.num_envs = 1

        # DQN family
        self.buffer_size = 100_000
        self.batch_size = 32
        self.train_freq = 4
        self.learn_start = 5_000
        self.eps_start = 1.0
        self.eps_end = 0.1
        self.eps_decay_steps = 100_000
        self.target_net = False
        self.target_sync = 1_000
        self.double = False
        self.dueling = False

        # PPO
        self.rollout = 512
        self.epochs = 10
        self.clip = 0.1
        self.lam = 0.95
        self.vf_coef = 1.0
        self.ent_coef = 0.01

        # logging
        self.log_every = 1_000
        self.policy_every = 5_000
        self.policy_states = 256

        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError(f"unknown config key {k!r}")
            setattr(self, k, v)

    def as_dict(self):
        return {k: v for k, v in vars(self).items()}


# The registry is what makes this one-click. Adding an algorithm to the
# benchmark means adding a line here, nothing else.
ALGORITHMS = {
    "ppo": dict(
        agent="ppo",
        label="PPO (task 6, discretised)",
        color="#7d3c98",
        overrides=dict(lr=2.5e-4, num_envs=4, rollout=512, batch_size=128),
    ),
    "dqn2013": dict(
        agent="dqn",
        label="DQN 2013, no target net (task 7)",
        color="#c0392b",
        overrides=dict(target_net=False, double=False, dueling=False),
    ),
    "dqn2015": dict(
        agent="dqn",
        label="DQN + target net (task 7)",
        color="#2471a3",
        overrides=dict(target_net=True, double=False, dueling=False),
    ),
    "ddqn": dict(
        agent="dqn",
        label="Double DQN (task 8)",
        color="#1e8449",
        overrides=dict(target_net=True, double=True, dueling=False),
    ),
    "dueling_ddqn": dict(
        agent="dqn",
        label="Dueling Double DQN (task 9)",
        color="#d68910",
        overrides=dict(target_net=True, double=True, dueling=True),
    ),
}


def build_config(name, **extra):
    if name not in ALGORITHMS:
        raise KeyError(f"unknown algorithm {name!r}. known: {list(ALGORITHMS)}")
    spec = ALGORITHMS[name]
    kw = dict(spec["overrides"])
    kw.update(extra)
    return Config(**kw)


def build_agent(name, cfg, device):
    spec = ALGORITHMS[name]
    rng = np.random.default_rng(cfg.seed)
    if spec["agent"] == "dqn":
        return DQNAgent(cfg, device, rng=rng)
    if spec["agent"] == "ppo":
        return PPOAgent(cfg, device, rng=rng)
    raise KeyError(spec["agent"])


def pick_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==========================================================================
# The runner
# ==========================================================================
def run_one(name, out_dir, steps=None, seed=0, device="auto", run_name=None,
            quiet=False, **extra):
    """
    Train one algorithm and write its logs. Knows nothing about which algorithm
    it is driving beyond the Agent interface.

    Writes three CSVs per run:
        {run}_episodes.csv   one row per finished episode
        {run}_updates.csv    one row per log_every steps, whatever learn() returned
        {run}_policy.csv     action distribution over a frozen state set
    """
    run_name = run_name or name
    cfg = build_config(name, seed=seed, **extra)
    if steps is not None:
        cfg.steps = steps

    dev = pick_device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = build_agent(name, cfg, dev)
    n_envs = agent.num_envs

    envs = [
        CarRacingDiscrete(seed=seed + i, img_stack=cfg.img_stack,
                          action_repeat=cfg.action_repeat)
        for i in range(n_envs)
    ]

    os.makedirs(out_dir, exist_ok=True)
    ck_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)

    ep_f = open(os.path.join(out_dir, f"{run_name}_episodes.csv"), "w", newline="")
    ep_w = csv.writer(ep_f)
    ep_w.writerow(["episode", "step", "reward", "length", "running_avg", "epsilon"])

    up_f = open(os.path.join(out_dir, f"{run_name}_updates.csv"), "w", newline="")
    up_w = csv.writer(up_f)
    up_cols = ["step", "loss", "mean_q", "mean_y", "entropy", "clip_frac",
               "epsilon", "steps_per_sec"]
    up_w.writerow(up_cols)

    pol_f = open(os.path.join(out_dir, f"{run_name}_policy.csv"), "w", newline="")
    pol_w = csv.writer(pol_f)
    pol_w.writerow(["step"] + [f"p_action{i}" for i in range(len(DISCRETE_ACTIONS))])

    # A frozen set of states for the policy figure, collected under the random
    # policy before learning starts. Same states for every algorithm at a given
    # seed, so the action distributions are comparable across the benchmark.
    policy_states = []

    states = []
    frames = []
    for e in envs:
        f, s = e.reset(seed=seed)
        frames.append(f)
        states.append(s)

    ep_reward = np.zeros(n_envs)
    ep_len = np.zeros(n_envs, dtype=int)
    recent = deque(maxlen=100)
    episode = 0
    best_avg = -1e9
    step = 0
    t0 = time.time()
    last_metrics = {}

    while step < cfg.steps:
        actions = agent.act(states, step)

        nf, ns, rw, tm, tr = [], [], [], [], []
        for i, e in enumerate(envs):
            f2, s2, r, term, trunc = e.step(int(actions[i]))
            nf.append(f2); ns.append(s2); rw.append(r); tm.append(term); tr.append(trunc)

        agent.observe(frames, actions, rw, tm, tr, ns)

        if len(policy_states) < cfg.policy_states:
            policy_states.append(states[0])

        ep_reward += np.array(rw)
        ep_len += 1
        step += n_envs

        for i in range(n_envs):
            if tm[i] or tr[i]:
                episode += 1
                recent.append(ep_reward[i])
                avg = float(np.mean(recent))
                ep_w.writerow([episode, step, round(float(ep_reward[i]), 2),
                               int(ep_len[i]), round(avg, 2),
                               round(float(getattr(agent, "eps", 0.0)), 4)])
                ep_f.flush()

                if not quiet and episode % 10 == 0:
                    el = time.time() - t0
                    print(f"[{run_name}] ep {episode:5d} | step {step:7d} "
                          f"| avg100 {avg:8.2f} | {step / max(el, 1e-9):5.1f} steps/s "
                          f"| {el / 60:.1f} min", flush=True)

                if len(recent) == recent.maxlen and avg > best_avg:
                    best_avg = avg
                    torch.save({**agent.state_dict(), "step": step, "episode": episode,
                                "best_avg": best_avg, "algo": name},
                               os.path.join(ck_dir, f"{run_name}_best.pt"))

                f2, s2 = envs[i].reset()
                nf[i], ns[i] = f2, s2
                ep_reward[i] = 0.0
                ep_len[i] = 0

        frames, states = nf, ns

        m = agent.learn(step)
        if m:
            last_metrics = m

        if step % cfg.log_every < n_envs and last_metrics:
            el = time.time() - t0
            up_w.writerow([step] + [
                round(last_metrics.get(c, float("nan")), 5) for c in up_cols[1:-1]
            ] + [round(step / max(el, 1e-9), 2)])
            up_f.flush()

        if step % cfg.policy_every < n_envs and len(policy_states) >= 32:
            probs = agent.policy_probs(policy_states)
            pol_w.writerow([step] + [round(float(p), 5) for p in probs])
            pol_f.flush()

    torch.save({**agent.state_dict(), "step": step, "episode": episode,
                "best_avg": best_avg, "algo": name},
               os.path.join(ck_dir, f"{run_name}_final.pt"))

    for f in (ep_f, up_f, pol_f):
        f.close()
    for e in envs:
        e.close()

    total = time.time() - t0
    summary = {
        "algo": name, "run_name": run_name, "steps": step, "episodes": episode,
        "best_avg100": round(best_avg, 2), "hours": round(total / 3600, 3),
        "steps_per_sec": round(step / max(total, 1e-9), 2),
        "config": cfg.as_dict(),
    }
    with open(os.path.join(out_dir, f"{run_name}_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    if not quiet:
        print(f"[{run_name}] done. {step} steps, {episode} episodes, "
              f"best avg100 {best_avg:.2f}, {total / 3600:.2f} h", flush=True)
    return summary


def _run_one_star(kwargs):
    """multiprocessing needs a module-level, picklable target."""
    try:
        return run_one(**kwargs)
    except Exception as e:               # one failure should not sink the benchmark
        import traceback
        traceback.print_exc()
        return {"algo": kwargs.get("name"), "error": str(e)}


def run_benchmark(names, out_dir, steps, seed=0, parallel=1, device="auto"):
    """
    Run several algorithms and collect their summaries.

    parallel > 1 forks worker processes. On one machine this contends for a
    single GPU and for memory (each DQN replay buffer is about 0.9 GB), so 2 is
    usually the practical ceiling locally. On a cluster the same benchmark is
    better expressed as one job per algorithm; see write_k8s_manifests.
    """
    jobs = [dict(name=n, out_dir=out_dir, steps=steps, seed=seed, device=device,
                 quiet=(parallel > 1)) for n in names]

    t0 = time.time()
    if parallel <= 1:
        results = [_run_one_star(j) for j in jobs]
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=parallel) as pool:
            results = pool.map(_run_one_star, jobs)

    elapsed = time.time() - t0
    with open(os.path.join(out_dir, "benchmark_summary.json"), "w") as fh:
        json.dump({"wall_clock_hours": round(elapsed / 3600, 3),
                   "parallel": parallel, "steps": steps, "seed": seed,
                   "results": results}, fh, indent=2, default=str)
    return results


def run_sweep(algo, param, values, out_dir, steps, seed=0, parallel=1, device="auto"):
    """
    Vary one hyperparameter and hold everything else fixed.

    This is the "refine the learning parameter" part of the task. It is the same
    runner, just with several configs of one algorithm instead of one config of
    several algorithms.
    """
    jobs = []
    for v in values:
        tag = f"{algo}_{param}{v:g}" if isinstance(v, float) else f"{algo}_{param}{v}"
        jobs.append(dict(name=algo, out_dir=out_dir, steps=steps, seed=seed,
                         device=device, run_name=tag, quiet=(parallel > 1),
                         **{param: v}))

    if parallel <= 1:
        results = [_run_one_star(j) for j in jobs]
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=parallel) as pool:
            results = pool.map(_run_one_star, jobs)

    with open(os.path.join(out_dir, f"sweep_{param}_summary.json"), "w") as fh:
        json.dump({"algo": algo, "param": param, "values": list(values),
                   "results": results}, fh, indent=2, default=str)
    return results


# ==========================================================================
# Figures
# ==========================================================================
# matplotlib and pandas are only needed for plotting, so they are imported
# lazily. A training-only container does not need them installed.
ACTION_NAMES = ["coast", "left", "right", "accelerate", "brake"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _load_runs(out_dir, names):
    import pandas as pd
    data = {}
    for n in names:
        d = {}
        for kind in ("episodes", "updates", "policy"):
            p = os.path.join(out_dir, f"{n}_{kind}.csv")
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                    d[kind] = df if not df.empty else None
                except Exception:
                    d[kind] = None
            else:
                d[kind] = None
        if d.get("episodes") is not None:
            data[n] = d
    return data


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _label(n):
    return ALGORITHMS.get(n, {}).get("label", n)


def _color(n, i=0):
    fallback = ["#c0392b", "#2471a3", "#1e8449", "#d68910", "#7d3c98", "#555555"]
    return ALGORITHMS.get(n, {}).get("color", fallback[i % len(fallback)])


def fig_rewards(data, path):
    """Reward, both ways the task asks for: running average and cumulative."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (n, d) in enumerate(data.items()):
        ep = d["episodes"]
        c = _color(n, i)
        axes[0].plot(ep["step"], ep["reward"], color=c, alpha=0.10, linewidth=0.6)
        axes[0].plot(ep["step"], ep["running_avg"], color=c, linewidth=2.0, label=_label(n))
        axes[1].plot(ep["step"], ep["reward"].cumsum(), color=c, linewidth=2.0, label=_label(n))

    _style(axes[0], "environment step", "episode return",
           "Return, 100-episode running average")
    _style(axes[1], "environment step", "cumulative return",
           "Cumulative return (area under the learning curve)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_loss(data, path):
    """
    Loss per algorithm.

    Separate panels rather than one axis, because a PPO surrogate loss and a DQN
    Huber loss are not the same quantity and putting them on shared axes would
    invite a comparison that means nothing.
    """
    plt = _plt()
    runs = list(data)
    if not runs:
        return
    ncol = min(3, len(runs))
    nrow = (len(runs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.8 * nrow), squeeze=False)

    for i, n in enumerate(runs):
        ax = axes[i // ncol][i % ncol]
        up = data[n]["updates"]
        if up is None or "loss" not in up:
            ax.set_visible(False)
            continue
        s = up["loss"].rolling(20, min_periods=1).mean()
        ax.plot(up["step"], s, color=_color(n, i), linewidth=1.6)
        if (s > 0).all():
            ax.set_yscale("log")
        _style(ax, "step", "loss (smoothed)", _label(n))

    for j in range(len(runs), nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_policies(data, path):
    """
    What each algorithm actually decided to do.

    Left: how the action distribution moved over training, stacked. Right: where
    it ended up, per algorithm. All measured on the same frozen states, so the
    columns are comparable.
    """
    plt = _plt()
    runs = [n for n in data if data[n]["policy"] is not None]
    if not runs:
        print("  (no policy logs, skipping the policy figure)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n_actions = len(ACTION_NAMES)

    # Left: the last algorithm's evolution is not enough, so show every
    # algorithm's accelerate share, the action that most separates them.
    for i, n in enumerate(runs):
        pol = data[n]["policy"]
        axes[0].plot(pol["step"], pol["p_action3"], color=_color(n, i),
                     linewidth=2.0, label=_label(n))
    _style(axes[0], "environment step", "share of states choosing accelerate",
           "Throttle preference over training")
    axes[0].legend(loc="best", frameon=False, fontsize=8)

    # Right: final distribution, grouped bars.
    width = 0.8 / len(runs)
    xs = np.arange(n_actions)
    for i, n in enumerate(runs):
        pol = data[n]["policy"]
        final = [float(pol[f"p_action{a}"].iloc[-1]) for a in range(n_actions)]
        axes[1].bar(xs + i * width - 0.4 + width / 2, final, width,
                    color=_color(n, i), label=_label(n), alpha=0.9)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(ACTION_NAMES)
    _style(axes[1], "", "share of frozen states", "Final policy")
    axes[1].legend(loc="best", frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_values(data, path):
    """Value estimates, carried over from task 8's held-out question."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 5))
    any_ = False
    for i, (n, d) in enumerate(data.items()):
        up = d["updates"]
        if up is None or "mean_q" not in up:
            continue
        ax.plot(up["step"], up["mean_q"].rolling(20, min_periods=1).mean(),
                color=_color(n, i), linewidth=1.8, label=_label(n))
        any_ = True
    if not any_:
        plt.close(fig)
        return
    _style(ax, "environment step", "mean predicted value",
           "Value estimates. For PPO this is the critic, for the DQNs it is mean Q.")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_sweep(out_dir, algo, param, path):
    """One panel for the hyperparameter study."""
    import pandas as pd
    plt = _plt()
    import glob
    runs = sorted(glob.glob(os.path.join(out_dir, f"{algo}_{param}*_episodes.csv")))
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, p in enumerate(runs):
        tag = os.path.basename(p).replace("_episodes.csv", "")
        df = pd.read_csv(p)
        ax.plot(df["step"], df["running_avg"], linewidth=2.0,
                label=tag.replace(f"{algo}_", ""))
    _style(ax, "environment step", "episode return, 100-ep average",
           f"{algo}: effect of {param}")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(out_dir, names, fig_dir=None):
    fig_dir = fig_dir or os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    data = _load_runs(out_dir, names)
    if not data:
        print("no runs found to plot")
        return data
    print(f"writing figures to {fig_dir}")
    fig_rewards(data, os.path.join(fig_dir, "01_rewards.png"))
    fig_loss(data, os.path.join(fig_dir, "02_loss.png"))
    fig_policies(data, os.path.join(fig_dir, "03_policies.png"))
    fig_values(data, os.path.join(fig_dir, "04_values.png"))
    return data


def print_table(out_dir, names):
    """The results table, printed so the write-up never has an unchecked number."""
    import pandas as pd
    rows = []
    for n in names:
        p = os.path.join(out_dir, f"{n}_episodes.csv")
        if not os.path.exists(p):
            continue
        ep = pd.read_csv(p)
        if ep.empty:
            continue
        sp = os.path.join(out_dir, f"{n}_summary.json")
        hours = sps = float("nan")
        if os.path.exists(sp):
            with open(sp) as fh:
                s = json.load(fh)
            hours, sps = s.get("hours", float("nan")), s.get("steps_per_sec", float("nan"))
        rows.append((n, len(ep), int(ep["step"].iloc[-1]), ep["running_avg"].max(),
                     ep["running_avg"].iloc[-1], ep["reward"].tail(100).mean(), hours, sps))

    if not rows:
        print("no results yet")
        return
    print("\n" + "=" * 100)
    print(f"{'algorithm':<16}{'episodes':>9}{'steps':>10}{'best avg100':>13}"
          f"{'final avg100':>14}{'last 100 ep':>13}{'hours':>8}{'steps/s':>9}")
    print("-" * 100)
    for r in sorted(rows, key=lambda x: -x[3]):
        print(f"{r[0]:<16}{r[1]:>9}{r[2]:>10,}{r[3]:>13.2f}{r[4]:>14.2f}"
              f"{r[5]:>13.2f}{r[6]:>8.2f}{r[7]:>9.1f}")
    print("=" * 100 + "\n")


# ==========================================================================
# Nautilus / Kubernetes
# ==========================================================================
K8S_JOB = """apiVersion: batch/v1
kind: Job
metadata:
  name: rl-bench-{safe}
spec:
  backoffLimit: 1
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: trainer
          image: {image}
          command: ["python", "rl_benchmark.py", "--algos", "{algo}",
                    "--steps", "{steps}", "--out-dir", "/results/{algo}",
                    "--device", "cuda", "--no-plot"]
          resources:
            requests: {{memory: 8Gi, cpu: "2", nvidia.com/gpu: "1"}}
            limits:   {{memory: 16Gi, cpu: "4", nvidia.com/gpu: "1"}}
          volumeMounts:
            - {{name: results, mountPath: /results}}
      volumes:
        - name: results
          persistentVolumeClaim:
            claimName: {pvc}
"""


def write_k8s_manifests(names, steps, out_dir, image, pvc="rl-benchmark-results"):
    """
    One Job per algorithm, which is what "benchmark them at the same time"
    actually means on a cluster. Each lands on its own GPU and writes into a
    shared volume, so the figures can be built from all of them afterwards.

    These are generated but untested; I have no cluster to submit them to.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for n in names:
        p = os.path.join(out_dir, f"job-{n.replace('_', '-')}.yaml")
        with open(p, "w") as fh:
            fh.write(K8S_JOB.format(safe=n.replace("_", "-"), algo=n,
                                    steps=steps, image=image, pvc=pvc))
        written.append(p)
    print(f"wrote {len(written)} manifests to {out_dir}")
    print("submit with:  kubectl apply -f " + out_dir)
    return written


# ==========================================================================
# CLI
# ==========================================================================
def main():
    p = argparse.ArgumentParser(
        description="one-click benchmark for the task 6-9 RL algorithms")
    p.add_argument("--all", action="store_true", help="run every registered algorithm")
    p.add_argument("--algos", nargs="*", default=None,
                   help=f"subset to run. choices: {list(ALGORITHMS)}")
    p.add_argument("--steps", type=int, default=200_000,
                   help="environment transitions per algorithm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--device", default="auto")
    p.add_argument("--parallel", type=int, default=1,
                   help="algorithms at a time. 2 is usually the local ceiling")

    p.add_argument("--sweep", default=None, metavar="PARAM",
                   help="sweep one parameter instead of benchmarking, e.g. lr")
    p.add_argument("--sweep-algo", default="ddqn")
    p.add_argument("--sweep-values", nargs="*", type=float,
                   default=[1e-4, 2.5e-4, 5e-4])
    p.add_argument("--sweep-steps", type=int, default=75_000)

    p.add_argument("--plot-only", action="store_true", help="redraw from existing CSVs")
    p.add_argument("--no-plot", action="store_true", help="train without plotting")
    p.add_argument("--k8s", default=None, metavar="IMAGE",
                   help="write Nautilus Job manifests for this image and exit")
    args = p.parse_args()

    names = list(ALGORITHMS) if (args.all or not args.algos) else args.algos
    for n in names:
        if n not in ALGORITHMS:
            p.error(f"unknown algorithm {n!r}. known: {list(ALGORITHMS)}")

    if args.k8s:
        write_k8s_manifests(names, args.steps, os.path.join(args.out_dir, "k8s"),
                            image=args.k8s)
        return

    if args.plot_only:
        make_figures(args.out_dir, names)
        print_table(args.out_dir, names)
        return

    if args.sweep:
        print(f"sweeping {args.sweep} over {args.sweep_values} on {args.sweep_algo}")
        run_sweep(args.sweep_algo, args.sweep, args.sweep_values, args.out_dir,
                  steps=args.sweep_steps, seed=args.seed, parallel=args.parallel,
                  device=args.device)
        if not args.no_plot:
            fig_dir = os.path.join(args.out_dir, "figures")
            os.makedirs(fig_dir, exist_ok=True)
            fig_sweep(args.out_dir, args.sweep_algo, args.sweep,
                      os.path.join(fig_dir, f"05_sweep_{args.sweep}.png"))
        return

    print(f"benchmarking {names}")
    print(f"{args.steps:,} steps each, seed {args.seed}, parallel {args.parallel}, "
          f"device {pick_device(args.device)}")
    run_benchmark(names, args.out_dir, args.steps, seed=args.seed,
                  parallel=args.parallel, device=args.device)

    if not args.no_plot:
        make_figures(args.out_dir, names)
    print_table(args.out_dir, names)


if __name__ == "__main__":
    main()
