"""
DQN from scratch, applied to a discretized CarRacing-v3.

This is a direct implementation of Mnih et al. 2013, "Playing Atari with Deep
Reinforcement Learning" (arXiv:1312.5602). No stable-baselines3, no RL library.
Only PyTorch, NumPy and Gymnasium.

The pieces of the paper that show up here:
  - Algorithm 1: epsilon-greedy action selection, store the transition, then
    sample a random minibatch from replay and take one gradient step
  - experience replay, Section 4, the thing that makes the whole method work
  - the Q-learning target r + gamma * max_a' Q(s', a'), Equation 3
  - the epsilon schedule, annealed from 1.0 to 0.1 over the early frames
  - the convolutional architecture from Section 4.1, resized for 96x96 input

Two deliberate deviations from the paper, both discussed in the write-up:

  1. No reward clipping. The paper clips every reward to {-1, 0, +1} because it
     wants one set of hyperparameters to work across seven different Atari games
     with wildly different scoring. CarRacing pays -0.1 per frame and +1000/N per
     track tile visited. Clipping would map "visited a tile" and "idled for one
     frame" to the same magnitude and destroy the only signal that distinguishes
     driving from sitting still. So rewards are passed through unclipped.

  2. The 2013 paper has no target network. That is the point of this experiment,
     not an oversight. Run it as published with the default settings, then pass
     --target-net to get the Nature 2015 fix, and compare the two.

Run:
    python dqn_carracing.py --steps 400000 --run-name dqn2013
    python dqn_carracing.py --steps 400000 --run-name dqn2015 --target-net
    python dqn_carracing.py --eval results/checkpoints/dqn2013_best.pt --episodes 5
"""

import argparse
import csv
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env_and_replay import (
    DISCRETE_ACTIONS,
    CarRacingDiscrete,
    ReplayBuffer,
    epsilon_at,
)


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------
class QNetwork(nn.Module):
    """
    The architecture from Section 4.1, with the input resized from the paper's
    84x84 to CarRacing's native 96x96 so no rescaling step is needed.

        conv 16 filters, 8x8, stride 4   ->  23x23
        conv 32 filters, 4x4, stride 2   ->  10x10
        fully connected, 256 units
        fully connected, one output per action

    One output unit per action is the trick from Section 3: a single forward pass
    gives Q for every action at once, so the max over actions costs nothing extra.
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
        # uint8 [0, 255] -> float [-1, 1], matching the task 6 scaling so the two
        # runs see the same inputs.
        x = x.float() / 128.0 - 1.0
        return self.head(self.features(x))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def open_csv(path, header, resume):
    exists = os.path.exists(path)
    f = open(path, "a" if resume and exists else "w", newline="")
    w = csv.writer(f)
    if not (resume and exists):
        w.writerow(header)
    return f, w


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(args):
    device = pick_device(args.device)
    print(f"device: {device}")
    print(f"target network: {'ON (Nature 2015)' if args.target_net else 'OFF (2013 paper as published)'}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env = CarRacingDiscrete(
        seed=args.seed, img_stack=args.img_stack, action_repeat=args.action_repeat
    )
    n_actions = len(DISCRETE_ACTIONS)

    online = QNetwork(args.img_stack, n_actions).to(device)
    target = None
    if args.target_net:
        target = QNetwork(args.img_stack, n_actions).to(device)
        target.load_state_dict(online.state_dict())
        target.eval()

    # The paper uses RMSProp. Adam would probably work better, but the point of
    # this week is to reproduce what is written down.
    optimizer = torch.optim.RMSprop(
        online.parameters(), lr=args.lr, alpha=0.95, eps=0.01
    )

    buffer = ReplayBuffer(args.buffer_size, args.img_stack)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    global_step = 0
    episode = 0
    best_avg = -1e9
    start_time = time.time()

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        online.load_state_dict(ckpt["online"])
        if target is not None and "target" in ckpt:
            target.load_state_dict(ckpt["target"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt["global_step"]
        episode = ckpt["episode"]
        best_avg = ckpt.get("best_avg", -1e9)
        print(f"resumed from {args.resume} at step {global_step}, episode {episode}")

    ep_file, ep_writer = open_csv(
        os.path.join(args.out_dir, f"{args.run_name}_episodes.csv"),
        ["episode", "step", "reward", "length", "running_avg", "epsilon"],
        args.resume,
    )
    up_file, up_writer = open_csv(
        os.path.join(args.out_dir, f"{args.run_name}_updates.csv"),
        ["step", "loss", "mean_q", "max_q", "epsilon", "buffer", "steps_per_sec"],
        args.resume,
    )

    recent = deque(maxlen=100)
    frame, state = env.reset(seed=args.seed)
    ep_reward = 0.0
    ep_len = 0
    loss_val = float("nan")
    mean_q = max_q = float("nan")

    stop_reason = "reached --steps"
    while global_step < args.steps:
        # Wall-clock budget. Throughput is hard to predict ahead of time, so this
        # guarantees the run ends when it is supposed to and leaves a usable
        # checkpoint and log rather than being killed halfway through a write.
        if args.max_hours and (time.time() - start_time) > args.max_hours * 3600:
            stop_reason = f"hit --max-hours ({args.max_hours} h)"
            break

        eps = epsilon_at(global_step, args.eps_start, args.eps_end, args.eps_decay_steps)

        # ---- act, epsilon-greedy (Algorithm 1) ----
        if global_step < args.learn_start or random.random() < eps:
            action = random.randrange(n_actions)
        else:
            with torch.no_grad():
                s = torch.as_tensor(state, device=device).unsqueeze(0)
                action = int(online(s).argmax(dim=1).item())

        next_frame, next_state, reward, terminated, truncated = env.step(action)

        # The frame that was *observed* going into this decision is `frame`, so
        # that is what gets stored alongside the action taken in it.
        buffer.add(
            frame, action, reward,
            terminal=terminated,
            boundary=terminated or truncated,
        )

        ep_reward += reward
        ep_len += 1
        global_step += 1
        frame, state = next_frame, next_state

        if terminated or truncated:
            episode += 1
            recent.append(ep_reward)
            avg = float(np.mean(recent))
            ep_writer.writerow([
                episode, global_step, round(ep_reward, 2), ep_len,
                round(avg, 2), round(eps, 4),
            ])
            ep_file.flush()

            if episode % 10 == 0:
                elapsed = time.time() - start_time
                sps = global_step / max(elapsed, 1e-9)
                print(
                    f"ep {episode:5d} | step {global_step:8d} | reward {ep_reward:8.2f} "
                    f"| avg100 {avg:8.2f} | eps {eps:.3f} | q {mean_q:7.2f} "
                    f"| {sps:6.1f} steps/s | {elapsed/60:.1f} min"
                )

            # Only start tracking "best" once the buffer is warm and the average
            # is over a full window, otherwise the first lucky episode wins.
            if len(recent) == recent.maxlen and avg > best_avg:
                best_avg = avg
                torch.save(
                    {
                        "online": online.state_dict(),
                        "target": target.state_dict() if target else None,
                        "optimizer": optimizer.state_dict(),
                        "global_step": global_step,
                        "episode": episode,
                        "best_avg": best_avg,
                        "args": vars(args),
                    },
                    os.path.join(ckpt_dir, f"{args.run_name}_best.pt"),
                )

            frame, state = env.reset()
            ep_reward = 0.0
            ep_len = 0

        # ---- learn ----
        if global_step >= args.learn_start and global_step % args.train_freq == 0:
            states, actions, rewards, next_states, terminals = buffer.sample(
                args.batch_size, rng
            )

            states_t = torch.as_tensor(states, device=device)
            next_states_t = torch.as_tensor(next_states, device=device)
            actions_t = torch.as_tensor(actions, device=device)
            rewards_t = torch.as_tensor(rewards, device=device)
            terminals_t = torch.as_tensor(terminals, device=device, dtype=torch.float32)

            # Q(s, a) for the action that was actually taken
            q = online(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

            # y = r + gamma * max_a' Q(s', a'), Equation 3.
            #
            # With no target network this max comes from the network being
            # updated, so the target moves the instant the weights do. That is
            # exactly the instability the 2015 paper fixed, and what this run is
            # designed to show.
            with torch.no_grad():
                bootstrap_net = target if target is not None else online
                next_q = bootstrap_net(next_states_t).max(dim=1).values
                y = rewards_t + args.gamma * next_q * (1.0 - terminals_t)

            # Huber rather than squared error. The paper's own follow-up notes
            # squared error blows up when rewards are unclipped, which they are
            # here by design.
            loss = F.smooth_l1_loss(q, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), args.max_grad_norm)
            optimizer.step()

            loss_val = float(loss.item())
            with torch.no_grad():
                mean_q = float(q.mean().item())
                max_q = float(q.max().item())

            if target is not None and global_step % args.target_sync == 0:
                target.load_state_dict(online.state_dict())

            if global_step % args.log_every == 0:
                elapsed = time.time() - start_time
                up_writer.writerow([
                    global_step, round(loss_val, 5), round(mean_q, 4), round(max_q, 4),
                    round(eps, 4), len(buffer),
                    round(global_step / max(elapsed, 1e-9), 2),
                ])
                up_file.flush()

        if global_step % args.save_every == 0:
            torch.save(
                {
                    "online": online.state_dict(),
                    "target": target.state_dict() if target else None,
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "episode": episode,
                    "best_avg": best_avg,
                    "args": vars(args),
                },
                os.path.join(ckpt_dir, f"{args.run_name}_latest.pt"),
            )

    torch.save(
        {
            "online": online.state_dict(),
            "target": target.state_dict() if target else None,
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "episode": episode,
            "best_avg": best_avg,
            "args": vars(args),
        },
        os.path.join(ckpt_dir, f"{args.run_name}_final.pt"),
    )

    ep_file.close()
    up_file.close()
    env.close()

    total = time.time() - start_time
    print(
        f"\ndone ({stop_reason}). {global_step} steps over {episode} episodes in "
        f"{total/3600:.2f} h ({total/60:.1f} min). best avg100 {best_avg:.2f}"
    )


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(args):
    """Greedy rollouts from a checkpoint, no exploration, on unseen tracks."""
    device = pick_device(args.device)
    ckpt = torch.load(args.eval, map_location=device)
    online = QNetwork(args.img_stack, len(DISCRETE_ACTIONS)).to(device)
    online.load_state_dict(ckpt["online"])
    online.eval()

    env = CarRacingDiscrete(
        seed=args.seed + 10_000,
        img_stack=args.img_stack,
        action_repeat=args.action_repeat,
    )

    scores = []
    for i in range(args.episodes):
        _, state = env.reset(seed=args.seed + 10_000 + i)
        total = 0.0
        done = False
        while not done:
            with torch.no_grad():
                s = torch.as_tensor(state, device=device).unsqueeze(0)
                action = int(online(s).argmax(dim=1).item())
            _, state, reward, terminated, truncated = env.step(action)
            total += reward
            done = terminated or truncated
        scores.append(total)
        print(f"  eval episode {i+1}: {total:.2f}")

    env.close()
    print(f"\nmean {np.mean(scores):.2f}  std {np.std(scores):.2f}  over {len(scores)} episodes")
    return scores


# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="DQN from scratch on discretized CarRacing-v3")
    p.add_argument("--steps", type=int, default=400_000,
                   help="agent decisions, not environment frames")
    p.add_argument("--max-hours", type=float, default=None,
                   help="wall-clock budget; stops cleanly and saves whatever it has")
    p.add_argument("--run-name", type=str, default="dqn2013")
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--img-stack", type=int, default=4, help="paper used 4")
    p.add_argument("--action-repeat", type=int, default=4, help="paper used 4")

    p.add_argument("--buffer-size", type=int, default=100_000,
                   help="paper used 1e6 frames; 1e5 here to fit in memory")
    p.add_argument("--batch-size", type=int, default=32, help="paper used 32")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--train-freq", type=int, default=4,
                   help="gradient steps happen every N agent decisions")
    p.add_argument("--learn-start", type=int, default=5_000,
                   help="pure random acting until the buffer has this many transitions")
    p.add_argument("--max-grad-norm", type=float, default=10.0)

    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.1, help="paper annealed to 0.1")
    p.add_argument("--eps-decay-steps", type=int, default=100_000)

    p.add_argument("--target-net", action="store_true",
                   help="add the Nature 2015 target network (off = 2013 paper as published)")
    p.add_argument("--target-sync", type=int, default=1_000,
                   help="copy online weights into the target every N steps")

    p.add_argument("--log-every", type=int, default=1_000)
    p.add_argument("--save-every", type=int, default=25_000)
    p.add_argument("--resume", type=str, default=None)

    p.add_argument("--eval", type=str, default=None, help="path to a checkpoint")
    p.add_argument("--episodes", type=int, default=5)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)
